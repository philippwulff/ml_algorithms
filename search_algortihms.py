from collections import defaultdict, deque
# Search algorithms


class Node:
    def __init__(self, val, neighbors: list):
        self.val = val
        self.neighbors = neighbors


def dfs(head: Node):
    """
    Depth first search using iteration.
    -> uses a stack (last in first out; LIFO)
    """

    stack = [head]
    visited = [head]

    while stack:
        current = stack.pop()

        for neigh in current.neighbors:

            if neigh not in visited:
                # do something
                print(neigh)
                stack.append(neigh)
                visited.append(neigh)

    return visited


def dfs_rec(head: Node, visited=[]):
    """
    Depth first search using recursion.
    """

    if head not in visited:
        visited.append(head)

        for neigh in head.neighbors:
            if neigh not in visited:
                # do something
                print(neigh)
                visited.append(neigh)
                dfs_rec(neigh, visited)

    return visited


def bfs(head):
    """
    Breadth first search
    -> uses a queue (first in first out; FIFO)
    """

    queue = deque(head)

    return
