use super::*;

type Path = (isize, Vec<usize>);
type Paths = Vec<Path>;

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

pub fn find_path<F1, F2>(width: usize,
                         height: usize,
                         start: usize,
                         branch: bool,
                         max_cost: usize,
                         reward_func: F1,
                         cost_func: F2) -> (Paths, Vec<usize>)
    where F1: Fn(usize) -> isize, F2: Fn(usize) -> usize
{
    let size = width * height;

    let mut neighbor_search = vec![Vec::new(); max_cost + 1];
    let mut n = 0;
    let mut queue = Vec::new();
    let mut path_reward = vec![0; size];
    let mut path_cost = vec![usize::MAX; size];
    let mut parents = vec![usize::MAX; size];
    let mut children = vec![Vec::new(); size];

    path_cost[start] = cost_func(start);
    path_reward[start] = reward_func(start);
    neighbor_search[0].push(start);
    let mut curr_cost = 0;

    loop {
        while n <= max_cost && neighbor_search[n].is_empty() {
            n += 1;
        }

        if n > max_cost {
            break;
        }

        queue.clear();

        for tile in &neighbor_search[n] {
            queue.extend(get_neighbors(width, height, *tile).into_iter());
        }

        neighbor_search[n].clear();

        for tile in queue.iter().copied() {
            let reward = reward_func(tile);
            let cost = cost_func(tile);

            if cost > max_cost || path_cost[tile] != usize::MAX {
                continue;
            }

            let mut best_neighbor = usize::MAX;

            for n in get_neighbors(width, height, tile) {
                if best_neighbor > size ||
                    path_cost[n] < path_cost[best_neighbor] ||
                    (path_cost[n] == path_cost[best_neighbor] &&
                     path_reward[n] > path_reward[best_neighbor])
                {
                    best_neighbor = n;
                }
            }

            if best_neighbor >= size {
                continue;
            }

            path_reward[tile] = path_reward[best_neighbor] + reward;
            path_cost[tile] = path_cost[best_neighbor] + cost;
            parents[tile] = best_neighbor;
            children[best_neighbor].push(tile);

            if path_cost[tile] <= max_cost {
                neighbor_search[path_cost[tile]].push(tile);
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

    let mut paths = vec![(0, Vec::new()); max_cost + 1];
    paths[cost_func(start)] = (reward_func(start), vec![start]);

    stack.push((paths, start, 0));

    while let Some((mut paths, tile, child)) = stack.pop() {
        if path_cost[tile] > max_cost {
            continue;
        }
        if paths.is_empty() {
            paths = vec![(0, Vec::new()); max_cost + 1];
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

pub fn debug_print_reward(width: usize, array: Vec<isize>) {
    for y in 0..array.len() / width {
        for x in 0..width {
            let i = x + y * width;

            print!("{} ", array[i]);
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

use std::collections::VecDeque;

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
            if out[n] != usize::MAX && map[n] >= 0 {
                out[n] = out[tile] + 1;
                queue.push_back(n);
            }
        }
    }

    return out;
}

pub fn nearest(width: usize, distance: &Vec<usize>, mut loc: usize) -> usize {
    let height = distance.len() / width;

    loop {
        let mut best = loc;

        for n in get_neighbors(width, height, loc) {
            if distance[n] < distance[best] {
                best = n;
            }
        }

        if best == loc {
            return loc;
        } else {
            loc = best;
        }
    }
}
