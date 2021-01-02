fn get_neighbors(width: usize, height: usize, loc: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let size = width * height;

    if loc >= width {
        out.push(loc - width);
    }
    if loc + width < size {
        out.push(loc + width);
    }
    if loc > 0 && loc / width == (loc - 1) / width {
        out.push(loc - 1);
    }
    if loc < size && loc / width == (loc + 1) / width {
        out.push(loc + 1);
    }

    out
}

type Path = (usize, Vec<usize>);
type Paths = Vec<(usize, Vec<usize>)>;

fn better_path(p1: &mut Path, p2: Path) {
    if p2.0 > p1.0 {
        *p1 = p2;
    }
}

fn combine_simple(_: usize, _: usize, paths1: &mut Paths, paths2: Paths) {
    for (p1, p2) in paths1.iter_mut().zip(paths2.into_iter()) {
        better_path(p1, p2);
    }
}

fn combine_branch(reward: usize, cost: usize, paths1: &mut Paths, paths2: Paths) {
    let len = paths1.len();

    for i in cost..len {
        for j in cost + 1..cost + len - i {
            let k = i + j - cost;
            let path = (paths1[i].0 + paths2[j].0 - reward, paths1[i].1.iter().copied().chain(paths2[j].1.iter().copied()).collect());

            better_path(&mut paths1[k], path);
        }
    }

    combine_simple(0, 0, paths1, paths2);
}

use std::collections::VecDeque;

pub fn find_path<F>(width: usize,
                    height: usize,
                    start: usize,
                    branch: bool,
                    max_cost: usize,
                    reward_func: F,
                    cost_func: F) -> Vec<(usize, Vec<usize>)>
    where F: Fn(usize) -> usize
{
    let size = width * height;

    let mut queue = VecDeque::new();
    let mut path_reward = vec![0; size];
    let mut path_cost = vec![usize::MAX; size];
    let mut parents = vec![usize::MAX; size];
    let mut children = vec![Vec::new(); size];

    queue.push_back(start);
    let mut curr_cost = 0;

    while let Some(tile) = queue.pop_front() {
        let reward = reward_func(tile);
        let cost = cost_func(tile);

        if cost > max_cost || path_cost[tile] != usize::MAX {
            continue;
        }

        let mut best_neighbor = usize::MAX;

        for n in get_neighbors(width, height, tile) {
            if best_neighbor > size ||
                cost_func(n) < cost_func(best_neighbor) ||
                (cost_func(n) == cost_func(best_neighbor) &&
                reward_func(n) > reward_func(best_neighbor))
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

        if path_cost[tile] < max_cost {
            for n in get_neighbors(width, height, tile) {
                queue.push_back(n);
            }
        }
    }

    // best_paths: (reward, starting points), tile, child_index
    let mut stack: Vec<(Vec<(usize, Vec<usize>)>, usize, usize)> = Vec::new();
    let combine =
        if branch {
            combine_branch
        } else {
            combine_simple
        };

    let mut paths = vec![(0, Vec::new()); max_cost + 1];
    paths[0] = (reward_func(start), vec![start]);

    stack.push((paths, start, 0));

    while let Some((mut paths, tile, child)) = stack.pop() {
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

    stack.pop().unwrap().0
}
