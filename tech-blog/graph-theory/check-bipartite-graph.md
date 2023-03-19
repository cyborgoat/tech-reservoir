---
title: Check whether a graph is bipartite
summary: This article describes what is bipartite graph and tells you how to determine if a graph is bipartite.
author: Junxiao Guo
date: 2023-03-15
tags:
    - graph-theory
---

A Bipartite Graph is a graph whose vertices can be divided into two independent sets, $U$ and $V$ such that every edge $(u, v)$ either connects a vertex from $U$ to $V$ or from $V$ to $U$. In other words, for every edge $(u, v)$, either $u$ belongs to $U$ and $v$ to $V$, or $u$ belongs to $V$ and $v$ to $U$,also, there is no edge that connects vertices of same set.

![bipartite-graph](https://dsm01pap004files.storage.live.com/y4mtC_mu7NmRP7dJnv4hYJ0oAA9ruGnRIZiH4r5CHxEkYLUJNwUqbg8gLNTlcVDN2dkd8dnFBDNa2DpUYaGzr4pCPPBkP_MLwnvpROhyeTvIS1Wv5s3vAWPKdwoz-jHY8vDUBSdyMSsGX-xxoWjyUWXi1UnR85TTZpp-sU1LnE13B8OTBDvRUvbt5u9B3ggeYUR?width=320&height=158&cropmode=none)

The algorithm to check for the bipartiteness of a graph is like creating a fascinating canvas of colors. 
1. We have an array called color\[\] which stores 0 or 1 for every node. 
2. Then we call the function DFS to traverse the graph starting from any node. 
3. For each node visited, we assign !color\[v\] to color\[u\] if u has not been visited before. 
4. We then call DFS again to visit nodes connected to u. If at any point, the color of u is equal to the color of v, then the node is not bipartite. 
5. Finally, we modify the DFS function such that it returns a boolean value at the end.


Let's define a graph as an 2D array, where graph\[u\] is an array of nodes that node u is adjacent to.

Below is the implementation of the above approach: 


```python
from typing import List
def is_bipartite(graph: List[List[int]]) -> bool:
        UNCOLORED,RED,BLUE = 0,1,2
        n = len(graph)
        status = [UNCOLORED] * n
        valid = True

        def dfs(node:int, color:int):
            nonlocal valid
            status[node] = color
            neighbor_color = (RED if color==BLUE else BLUE)
            for neighbor in graph[node]:
                if status[neighbor] == UNCOLORED:
                    dfs(neighbor,neighbor_color)
                    if valid == False:
                        return 
                elif status[neighbor] != neighbor_color:
                    valid = False
                    return

        for i in range(n):
            if status[i] == UNCOLORED:
                dfs(i,RED)
                if not valid:
                    break

        return valid

```


```python
graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
result = is_bipartite(graph)
print(result)
```

    False



```python
graph = [[1,3],[0,2],[1,3],[0,2]]
result = is_bipartite(graph)
print(result)
```

    True



```python

```
