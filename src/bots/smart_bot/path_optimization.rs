use super::*;

pub type Path = (isize, Vec<usize>);
pub type Paths = Vec<Path>;

const COST_WEIGHT: f64 = 0.2;

fn better_path(p1: &mut Path, p2: Path) {
    if p2.0 > p1.0 {
        *p1 = p2;
    }
}

fn combine_simple(_: isize, _: usize, paths1: &mut Paths, paths2: Paths) {
    for (p1, p2) in paths1.iter_mut().zip(paths2.into_iter()) {
        better_path(p1, p2);
    }
}

fn combine_branch(reward: isize, cost: usize, paths1: &mut Paths, paths2: Paths) {
    let len = paths1.len();
    let mut out = paths1.clone();

    for i in cost + 1..len {
        for j in cost + 1..cost + len - i {
            let k = i + j - cost;
            let path = (paths1[i].0 + paths2[j].0 - reward, paths1[i].1.iter().copied().chain(paths2[j].1.iter().copied()).collect());

            better_path(&mut out[k], path);
        }
    }

    combine_simple(0, 0, &mut out, paths2);
    *paths1 = out;
}

use std::collections::{VecDeque, BinaryHeap};

pub fn find_path<F1>(width: usize,
                     start: usize,
                     branch: bool,
                     max_cost: usize,
                     reward_func: F1,
                     obstacles: &Vec<bool>) -> (Paths, Vec<usize>)
    where F1: Fn(usize) -> isize
{
    let size = obstacles.len();
    let height = size / width;

    let mut path_cost = vec![usize::MAX; size];
    let mut path_reward = vec![isize::MIN; size];
    let mut parents = vec![usize::MAX; size];
    let mut children = vec![Vec::new(); size];

    let mut queue: BinaryHeap<(isize, usize)> = BinaryHeap::new();
    let mut avg_reward = reward_func(start) as f64;
    let mut n_searched = 1.;

    path_reward[start] = reward_func(start);
    path_cost[start] = 0;
    queue.push((0, start));

    while let Some((_, loc)) = queue.pop() {
        if path_cost[loc] > max_cost {
            continue;
        }

        for n in get_neighbors(width, height, loc) {
            if !obstacles[n] && path_reward[n] == isize::MIN {
                let reward = reward_func(n);

                path_reward[n] = path_reward[loc] + reward;
                path_cost[n] = path_cost[loc] + 1;

                children[loc].push(n);
                parents[n] = loc;

                avg_reward =
                    avg_reward * (n_searched - 1.) / n_searched +
                    (reward as f64).abs() / n_searched;

                n_searched += 1.;

                let h =
                    path_reward[n] -
                    (COST_WEIGHT *
                     avg_reward *
                     (max_cost - path_cost[n]) as f64 *
                     path_cost[n] as f64) as isize;

                queue.push((h, n));
            }
        }
    }

    // best_paths: (reward, starting points), tile, child_index
    let mut stack: Vec<(Paths, usize, usize)> = Vec::new();
    let combine =
        if branch {
            combine_branch
        } else {
            combine_simple
        };

    let mut paths = vec![(0, Vec::new()); max_cost];
    paths[0] = (reward_func(start), vec![start]);

    stack.push((paths, start, 0));

    while let Some((mut paths, tile, child)) = stack.pop() {
        if path_cost[tile] >= max_cost {
            continue;
        }
        if paths.is_empty() {
            paths = vec![(0, Vec::new()); max_cost];
            paths[path_cost[tile]] = (path_reward[tile], vec![tile]);
        }
        if child < children[tile].len() {
            stack.push((paths, tile, child + 1));
            stack.push((Vec::new(), children[tile][child], 0));
        } else if tile == start {
            stack.push((paths, tile, child));
            break;
        } else if let Some((old_paths, i, _)) = stack.last_mut() {
            combine(path_reward[*i], path_cost[*i], old_paths, paths);
        }
    }

    (stack.pop().unwrap().0, parents)
}

pub fn debug_print_parents(width: usize, parents: Vec<usize>) {
    for y in 0..parents.len() / width {
        for x in 0..width {
            let i = x + y * width;
            let d = i as isize - parents[i] as isize;

            if d == -1 {
                print!("> ");
            } else if d == 1 {
                print!("< ");
            } else if d == width as isize {
                print!("^ ");
            } else if d == -(width as isize) {
                print!("v ");
            } else {
                print!(". ");
            }
        }
        println!();
    }
    println!();
}

pub fn debug_print_reward<T>(width: usize, array: Vec<T>)
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
pub struct PathTree {
    loc: usize,
    priority: isize,
    children: Vec<PathTree>
}

pub fn get_sequences(path: &Path, parents: &Vec<usize>) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); path.1.len()];

    for (i, l) in path.1.iter().enumerate() {
        let mut loc = *l;

        while loc != usize::MAX {
            out[i].push(loc);
            loc = parents[loc];
        }
    }

    out
}

impl PathTree {
    fn new(loc: usize) -> Self {
        Self {
            loc,
            priority: 0,
            children: Vec::new()
        }
    }

    pub fn from_path(path: &Path, parents: &Vec<usize>) -> Self {
        let paths = get_sequences(path, parents);

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

        out.children.into_iter().next().unwrap()
    }

    pub fn apply_priority<F>(&mut self, pri_func: &F) where F: Fn(usize) -> isize {
        for c in &mut self.children {
            c.apply_priority(pri_func);
        }

        self.priority = self.children.iter().map(|x| x.priority).sum();
        self.priority += pri_func(self.loc);

        self.children.sort_unstable_by_key(|x| x.priority);
    }

    pub fn serialize_inwards(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();

        for c in &self.children {
            out.extend(c.serialize_inwards().into_iter());
            out.push((c.loc, self.loc));
        }

        out
    }

    pub fn serialize_outwards(&self) -> Vec<(usize, usize)> {
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

pub fn nearest(width: usize, distance: &Vec<usize>, mut loc: usize)
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
