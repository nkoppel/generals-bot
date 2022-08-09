use super::*;

pub type Path = (f64, Vec<usize>);
pub type Paths = Vec<Path>;

fn combine_simple(_: f64, _: usize, paths1: &mut Paths, paths2: Paths) {
    for (p1, p2) in paths1.iter_mut().zip(paths2.into_iter()) {
        if p2.0 > p1.0 {
            *p1 = p2;
        }
    }
}

fn combine_branch(reward: f64, len: usize, paths1: &mut Paths, paths2: Paths) {
    let max_len = paths1.len();
    let mut out = paths1.clone();

    for i in len + 1..max_len {
        for j in len + 1..len + max_len - i {
            if !paths1[i].1.is_empty() && !paths2[j].1.is_empty() {
                let k = i + j - len;
                let reward2 = paths1[i].0 + paths2[j].0 - reward;

                if reward2 > out[k].0 {
                    out[k] = (reward2, paths1[i].1.iter().copied().chain(paths2[j].1.iter().copied()).collect());
                }
            }
        }
    }

    combine_simple(0., 0, &mut out, paths2);
    *paths1 = out;
}

use std::collections::VecDeque;

pub struct Pather<'a> {
    width: usize,
    reward: &'a Vec<f64>,
    obstacle: &'a Vec<bool>,
    path_len: Vec<usize>,
    path_reward: Vec<f64>,
    parent: Vec<usize>,
    children: Vec<Vec<usize>>,
}

impl<'a> Pather<'a> {
    pub fn new(width: usize, reward: &'a Vec<f64>, obstacle: &'a Vec<bool>)
        -> Self
    {
        let size = obstacle.len();

        Self {
            width,
            reward,
            obstacle,
            path_len: vec![usize::MAX; size],
            path_reward: vec![f64::NEG_INFINITY; size],
            parent: vec![usize::MAX; size],
            children: vec![Vec::new(); size],
        }
    }

    pub fn reset(&mut self) {
        let size = self.obstacle.len();

        self.path_len.fill(usize::MAX);
        self.path_reward.fill(f64::NEG_INFINITY);
        self.parent.fill(usize::MAX);
        self.children.fill(Vec::new());
    }

    fn is_ancestor(&self, parent: usize, mut child: usize) -> bool {
        while child != usize::MAX && child != parent {
            child = self.parent[child];
        }

        child == parent
    }

    fn update_decendents(&mut self, loc: usize) {
        for child in self.children[loc].clone() {
            self.path_len[child] = self.path_len[loc] + 1;
            self.path_reward[child] = self.path_reward[loc] + self.reward[child];

            self.update_decendents(child);
        }
    }

    fn neighbors(&self, loc: usize) -> Vec<usize> {
        let height = self.obstacle.len() / self.width;
        let mut out = get_neighbors(self.width, height, loc);

        out.retain(|l| !self.obstacle[*l]);
        out
    }

    fn path_quality(&self, loc: usize, max_len: usize, branch: bool) -> f64 {
        if self.path_len[loc] >= max_len {
            return f64::NEG_INFINITY;
        }

        if branch {
            self.path_reward[loc] as f64 / self.path_len[loc] as f64
        } else {
            self.path_reward[loc] as f64
        }
    }

    pub fn create_graph(&mut self,
                        start: usize,
                        max_len: usize,
                        refine_iters: usize,
                        branch: bool)
    {
        let mut queue = VecDeque::new();

        self.path_reward[start] = self.reward[start];
        self.path_len[start] = 0;

        queue.extend(self.neighbors(start));

        while let Some(loc) = queue.pop_front() {
            if self.path_len[loc] != usize::MAX {
                continue;
            }

            let mut best_parent = usize::MAX;
            let mut best_reward = f64::NEG_INFINITY;

            for parent in self.neighbors(loc) {
                if self.path_reward[parent] > best_reward {
                    best_parent = parent;
                    best_reward = self.path_reward[parent];
                }
            }

            if best_parent > self.parent.len() {
                continue;
            }

            self.parent[loc] = best_parent;
            self.children[best_parent].push(loc);

            self.path_reward[loc] = self.reward[loc] + best_reward;
            self.path_len[loc] = self.path_len[best_parent] + 1;

            if self.path_len[loc] < max_len {
                for neighbor in self.neighbors(loc) {
                    if self.path_len[neighbor] == usize::MAX {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        for _ in 0..refine_iters {
            let mut improved = false;
            queue.extend(self.neighbors(start));

            while let Some(loc) = queue.pop_front() {
                let children = self.children[loc].clone();
                let parent = self.parent[loc];
                let mut neighbors = self.neighbors(loc);

                if parent > self.parent.len() {
                    continue;
                }

                neighbors.retain(|n|
                   !children.contains(n)
                    && *n != parent
                    && self.path_quality(*n, max_len, branch)
                       > self.path_quality(parent, max_len, branch)
                    && self.path_len[*n] < max_len
                );

                neighbors.sort_by(|a, b|
                    self.path_quality(*a, max_len, branch)
                        .partial_cmp(&self.path_quality(*b, max_len, branch))
                        .unwrap()
                );

                for n in neighbors {
                    if !self.is_ancestor(loc, n) {
                        self.path_len[loc] = self.path_len[n] + 1;
                        self.path_reward[loc] = self.path_reward[n] + self.reward[loc];

                        let remove_ind = self.children[parent]
                            .iter()
                            .position(|x| *x == loc)
                            .unwrap();

                        self.children[n].push(loc);
                        self.children[parent].remove(remove_ind);
                        self.parent[loc] = n;

                        self.update_decendents(loc);

                        improved = true;
                        break;
                    }
                }

                queue.extend(children);
            }

            if !improved {
                break;
            }
        }
    }

    pub fn get_best_paths(&self, start: usize, max_len: usize, branch: bool)
        -> Paths
    {
        // best_paths: (reward, starting points), tile, child_index
        let mut stack: Vec<(Paths, usize, usize)> = Vec::new();
        let combine =
            if branch {
                combine_branch
            } else {
                combine_simple
            };

        let mut paths = vec![(0., Vec::new()); max_len + 1];
        paths[0] = (self.reward[start], vec![start]);

        stack.push((paths, start, 0));

        while let Some((mut paths, tile, child)) = stack.pop() {
            if self.path_len[tile] > max_len {
                continue;
            }
            if paths.is_empty() {
                paths = vec![(0., Vec::new()); max_len + 1];
                paths[self.path_len[tile]] = (self.path_reward[tile], vec![tile]);
            }
            if child < self.children[tile].len() {
                stack.push((paths, tile, child + 1));
                stack.push((Vec::new(), self.children[tile][child], 0));
            } else if tile == start {
                stack.push((paths, tile, child));
                break;
            } else if let Some((old_paths, i, _)) = stack.last_mut() {
                combine(self.path_reward[*i], self.path_len[*i], old_paths, paths);
            }
        }

        stack.pop().unwrap().0
    }

    pub fn get_moves(&self, path: &Path, inwards: bool)
        -> Vec<(usize, usize)>
    {
        let tree = PathTree::from_path(&path, &self.parent);

        if inwards {
            tree.serialize_inwards()
        } else {
            tree.serialize_outwards()
        }
    }

    pub fn debug_print_parents(&self) {
        for y in 0..self.parent.len() / self.width {
            for x in 0..self.width {
                let i = x + y * self.width;
                let d = i as i128 - self.parent[i] as i128;

                if d == -1 {
                    print!("> ");
                } else if d == 1 {
                    print!("< ");
                } else if d == self.width as i128 {
                    print!("^ ");
                } else if d == -(self.width as i128) {
                    print!("v ");
                } else {
                    print!(". ");
                }
            }
            println!();
        }
        println!();
    }

    pub fn debug_print(&self, field: &str) {
        match field {
            "reward"      => debug_print_2d(self.width, &self.reward),
            "path_reward" => debug_print_2d(self.width, &self.path_reward),
            "path_len"    => debug_print_2d(self.width, &self.path_len),
            _ => {}
        }
    }
}

pub fn debug_print_2d<T>(width: usize, array: &Vec<T>)
    where T: std::fmt::Display
{
    for y in 0..array.len() / width {
        for x in 0..width {
            let i = x + y * width;

            print!("{:5} ", array[i]);
        }
        println!();
    }
    println!();
}

#[derive(Clone, Debug)]
struct PathTree {
    loc: usize,
    children: Vec<PathTree>
}

fn get_sequences(path: &Path, parent: &Vec<usize>) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); path.1.len()];

    for (i, l) in path.1.iter().enumerate() {
        let mut loc = *l;

        while loc != usize::MAX {
            out[i].push(loc);
            loc = parent[loc];
        }
    }

    out
}

impl PathTree {
    fn new(loc: usize) -> Self {
        Self {
            loc,
            children: Vec::new()
        }
    }

    fn from_path(path: &Path, parent: &Vec<usize>) -> Self {
        let paths = get_sequences(path, parent);

        let mut out = Self::new(0);
        let mut pointer = &mut out;

        for path in paths {
            for p in path.iter().rev() {
                if let Some(pos) = pointer.children.iter().position(|x| *p == x.loc) {
                    pointer = &mut pointer.children[pos];
                } else {
                    pointer.children.push(Self::new(*p));
                    pointer = pointer.children.last_mut().unwrap();
                }
            }

            pointer = &mut out;
        }

        out.children.into_iter()
            .next()
            .unwrap_or(Self::new(0))
    }

    fn serialize_inwards(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();

        for c in &self.children {
            out.extend(c.serialize_inwards().into_iter());
            out.push((c.loc, self.loc));
        }

        out
    }

    fn serialize_outwards(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();

        for c in &self.children {
            out.push((self.loc, c.loc));
            out.extend(c.serialize_outwards().into_iter());
        }

        out
    }
}

// < 0: obstacle, > 0: object to measure distance from
pub fn min_distance(width: usize, map: &Vec<isize>) -> Vec<usize> {
    let height = map.len() / width;
    let mut queue = VecDeque::new();
    let mut out = vec![usize::MAX; map.len()];

    for i in 0..map.len() {
        if map[i] > 0 {
            out[i] = 0;
            queue.push_back(i);
        }
    }

    while let Some(tile) = queue.pop_front() {
        for n in get_neighbors(width, height, tile) {
            if out[n] == usize::MAX && map[n] >= 0 {
                out[n] = out[tile] + 1;
                queue.push_back(n);
            }
        }
    }

    return out;
}

pub fn nearest_zero(width: usize, distance: &Vec<usize>, mut loc: usize)
    -> Option<usize>
{
    let height = distance.len() / width;

    if distance[loc] == usize::MAX {
        return None;
    }

    loop {
        let mut best = loc;

        for n in get_neighbors(width, height, loc) {
            if distance[n] < distance[best] {
                best = n;
            }
        }

        if best == loc {
            return Some(loc);
        } else {
            loc = best;
        }
    }
}

pub fn nearest_zero_path(width: usize, distance: &Vec<usize>, mut loc: usize)
    -> Vec<usize>
{
    let height = distance.len() / width;

    if distance[loc] == usize::MAX {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(distance[loc]);

    loop {
        out.push(loc);
        let mut best = loc;

        for n in get_neighbors(width, height, loc) {
            if distance[n] < distance[best] {
                best = n;
            }
        }

        if best == loc {
            return out;
        } else {
            loc = best;
        }
    }
}

mod tests {
    extern crate test;

    use super::*;
    use test::Bencher;

    #[test]
    fn t_path_finder() {
        let reward = vec![
             0.,  0.,  0.,  0.,  0.,
             1.,  1.,  0.,  0.,  0.,
            -1.,  2.,  0.,  0.,  0.,
             1.,  1.,  0.,  0.,  0.,
             1.,  1.,  0.,  0.,  0.,
        ];
        let obstacle1 = vec![false; 25];
        let obstacle2 = vec![
            false, false, false, false, false,
            false, false, false, false, false,
            false, true , true , false, false,
            false, false, false, false, false,
            false, false, false, false, false,
        ];

        let mut pather = Pather::new(5, &reward, &obstacle1);

        pather.create_graph(0, 7, 1, true);
        let mut paths = pather.get_best_paths(0, 7, true);
        let mut moves = pather.get_moves(&paths[7], true);

        pather.debug_print_parents();
        pather.debug_print("path_reward");

        assert_eq!(paths[7].0, 8.);
        assert_eq!(moves.len(), 7);

        pather.reset();

        pather.create_graph(0, 7, 1, false);
        paths = pather.get_best_paths(0, 7, false);
        moves = pather.get_moves(&paths[7], true);

        assert_eq!(paths[7].0, 8.);
        assert_eq!(moves.len(), 7);

        pather = Pather::new(5, &reward, &obstacle2);

        pather.create_graph(0, 7, 1, true);
        paths = pather.get_best_paths(0, 7, true);
        moves = pather.get_moves(&paths[7], true);

        pather.debug_print_parents();

        assert_eq!(paths[7].0, 5.);
        assert_eq!(moves.len(), 7);

        let reward2 = vec![0.; 100];
        let obstacle3 = vec![false; 100];

        pather = Pather::new(10, &reward2, &obstacle3);
        pather.create_graph(0, 5, 1, true);

        pather.debug_print_parents();
        panic!();
    }

    #[bench]
    fn b_create_graph(b: &mut Bencher) {
        let mut rng = rand::thread_rng();
        let width = 41;
        let size = width * width;
        let start = size / 2 + width / 2;

        let obstacles = vec![false; size];
        let reward = std::iter::repeat_with(|| rng.gen::<f64>() * 200. - 100.)
            .take(size)
            .collect();

        let mut pather = Pather::new(width, &reward, &obstacles);

        b.iter(|| {
            pather.create_graph(start, 40, 1, true);
            let out = pather.parent[start + 1];
            pather.reset();
            out
        });
    }

    #[bench]
    fn b_get_best_paths_branching(b: &mut Bencher) {
        let mut rng = rand::thread_rng();
        let width = 41;
        let size = width * width;
        let start = size / 2 + width / 2;

        let obstacles = vec![false; size];
        let reward = std::iter::repeat_with(|| rng.gen::<f64>() * 200. - 100.)
            .take(size)
            .collect();

        let mut pather = Pather::new(width, &reward, &obstacles);

        pather.create_graph(start, 40, 1, true);

        b.iter(|| {
            pather.get_best_paths(start, 40, true);
        });
    }

    #[bench]
    fn b_get_best_paths_nonbranching(b: &mut Bencher) {
        let mut rng = rand::thread_rng();
        let width = 41;
        let size = width * width;
        let start = size / 2 + width / 2;

        let obstacles = vec![false; size];
        let reward = std::iter::repeat_with(|| rng.gen::<f64>() * 200. - 100.)
            .take(size)
            .collect();

        let mut pather = Pather::new(width, &reward, &obstacles);

        pather.create_graph(start, 40, 1, true);


        b.iter(|| {
            pather.get_best_paths(start, 40, false);
        });
    }

    #[bench]
    fn b_pather(b: &mut Bencher) {
        let mut rng = rand::thread_rng();
        let width = 41;
        let size = width * width;
        let start = size / 2 + width / 2;

        let obstacles = vec![false; size];
        let reward = std::iter::repeat_with(|| rng.gen::<f64>() * 200. - 100.)
            .take(size)
            .collect();

        b.iter(|| {
            let mut pather = Pather::new(width, &reward, &obstacles);

            pather.create_graph(start, 40, 1, true);
            let paths = pather.get_best_paths(start, 40, true);

            pather.get_moves(paths.last().unwrap(), true)
        });
    }
}
