
# input: given a list of 2d points   [(x_1, y_1) ,... (x_n, y_n)]
# output: ordered point sequence, when connecting consecutive points, it forms a polygon

def polygon_sequence(points: List[Tuple(int, int)]) -> List[Tuple(int, int)]:
    # return ...


"""

  x1 .----> . x2
       \  /
        / \
      x3 .---. x4

not polygon: x1, x2, x3, x4 (x2->x3, x4->x1 these 2 lines intersect)
polygon: x1, x2, x4, x3

convex hull
"""

    if len(points) <= 3:
        return 
    
    def calculate_centroids(points):
        sum_x = 0
        sum_y = 0
        n = len(points)
        
        for x, y in points:
            sum_x += x
            sum_y += y
            
        return (sum_x/n, sum_y/n)   
         
    
    def calculate_angle(point, centroid):
        
    
    # List of tuples [(angle, point)]
    point_with_angles = []
    
    # Find centroids
    centroid = calculate_centroids(points)
    
    # Calculate the angles of each point
    for point in points:
        angle = calculate_angle(point, centroid)
        point_with_angles.append((point, angle))
    
    # sort points by angle
    point_with_angles.sort(key = lambda x: x[1])
    
    # Extract the sorted poins
    res = []
    for p, a in point_with_angles:
        res.append(p)
        
    return 

// ======================== Question 2 =====================================

A = [a0, a1, a2, ..., a(n-1)]
B = [b0, b1, b2, ..., b(m-1)]
- we have: n <= m
- define: C = |a0 - b(k0)| + |a1 - b(k1)| + .... + |a(n) - b(k{n-1})|
- 0 <= k0 < k1 < .. < k{n-1} < m
Write a program to compute the smallest C given two arrays `A` and `B`? 

dp[n][m]

dp[i][j] meaning?

d[0..=2][0..=4] 


A' = [a0, a1]
B' = [b0, b1, b2, b3]

A' = [a0, a1]
B' = [b0, b1, b2, b3, b4]

dp[2][5] = min(dp )


dp[i][j] = MIN(dp[i-1][j-1] + abs(a[i] - b[j]), dp[i][j-1]);



# ---------------------------------------------------------------------------- #
#                              Another Interviewer                             #
# ---------------------------------------------------------------------------- #
; This function should turn the symbols `+' and `-' into unsigned and signed
; versions.
;
; The first argument is a list of descriptions of variables.  Each
; entry is a list containing the name and type.
;
; The second argument is a list of lines in the form
;   '(write <variable> <value>)
; <variable> is a name of a variable in the first argument.
; <value> is one of the following.
;   A number.
;   A variable name.
;   (<op> <value> <value>) where <op> is '+ or '-.
(define (infer-operations variables program)
  ; Fill
  	
  
  )

; Expected result when the input uses consistent types.
(equal?
 '((write dest1 (unsigned+ opu1 (unsigned* opu1 opu2)))
   (write dest2 (signed- ops1 1)))
 (infer-operations
  '((opu1 unsigned) (opu2 unsigned) (ops1 signed))
  '((write opu1 (+ opu1 (* opu1 opu2)))
    (write ops1 (- ops1 1)))))

; Expected result when the input mixes types.
(equal?
 #f
 (infer-operations
  '((opu1 unsigned) (opu2 unsigned) (ops1 signed)) 
  '((write opu1 (+ ops1 (* opu1 opu2)))
    (write ops1 (- ops1 1)))))



# A piece of program to read from standard input a stream of characters, when the input ends, it should 
# indicate if there is a palindrome
# Restriction (it needs to run in o(N) time and O(1) space)

f^0(x[0]) + f^1(x[1]) ...
f^(N)(x[0]) + f^(N-1)(x[1]) ...

def get() # Give me the next byte
def hash(char): # A hash function that is "symmetric"

  

curr_char = ""
hash_forward = 
hash_backward = 

for c in get():
	
  "abcba"
def is_palindrome(string: str) -> bool:

  start= 0
  end = len(str) -1
	for i in range(len(str)//2):
		if str[start] != str[end]:
    	return False
  return True
  
  
# ---------------------------------------------------------------------------- #
#                                 Tree Scatter                                 #
# ---------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt

def generate_trees(x_min, x_max, y_min, y_max):
  """
  To implement!
  (N, 2)
  """
  return np.random.random((100, 2)) * 1000

x_min, x_max, y_min, y_max = -1000, 1000, -1000, 1000

trees = generate_trees(x_min, x_max, y_min, y_max)

plt.gca().set_xlim((x_min, x_max))
plt.gca().set_ylim((y_min, y_max))
plt.scatter(trees[:, 0], trees[:, 1])


# ---------------------------------------------------------------------------- #
#                                 Weird Scheme                                 #
# ---------------------------------------------------------------------------- #
; This function should turn the symbols `+' and `-' into unsigned and signed
; versions.
;
; The first argument is a list of descriptions of variables.  Each
; entry is a list containing the name and type.
;
; The second argument is a list of lines in the form
;   '(write <variable> <value>)
; <variable> is a name of a variable in the first argument.
; <value> is one of the following.
;   A number.
;   A variable name.
;   (<op> <value> <value>) where <op> is '+ or '-.
(define (infer-operations variables program)
  ; Fill
  	
  
  )

; Expected result when the input uses consistent types.
(equal?
 '((write dest1 (unsigned+ opu1 (unsigned* opu1 opu2)))
   (write dest2 (signed- ops1 1)))
 (infer-operations
  '((opu1 unsigned) (opu2 unsigned) (ops1 signed))
  '((write opu1 (+ opu1 (* opu1 opu2)))
    (write ops1 (- ops1 1)))))

; Expected result when the input mixes types.
(equal?
 #f
 (infer-operations
  '((opu1 unsigned) (opu2 unsigned) (ops1 signed)) 
  '((write opu1 (+ ops1 (* opu1 opu2)))
    (write ops1 (- ops1 1)))))



# A piece of program to read from standard input a stream of characters, when the input ends, it should 
# indicate if there is a palindrome
# Restriction (it needs to run in o(N) time and O(1) space)

f^0(x[0]) + f^1(x[1]) ...
f^(N)(x[0]) + f^(N-1)(x[1]) ...

def get() # Give me the next byte
def hash(char): # A hash function that is "symmetric"

  

curr_char = ""
hash_forward = 
hash_backward = 

for c in get():
	
  "abcba"
def is_palindrome(string: str) -> bool:

  start= 0
  end = len(str) -1
	for i in range(len(str)//2):
		if str[start] != str[end]:
    	return False
  return True
  

# ---------------------------------------------------------------------------- #
#                             Zero Point Detection                             #
# ---------------------------------------------------------------------------- #


# Function to dertermine the preriod given an amplitude array and sampling period

# Determine half period

def calc_period(sampling_period, arr):
  	len(arr) > 3:
      return -1
  	
  	zero_point_candidtates = [] # (left_idx, right_idx)
    prev = 0
    arr = arr[1:]
    mode = True if arr[prev] > 0 False # Search negative
    
  	for i, a in enumerate(arr):
      if len(zero_point_candidtates) == 2:
        break
        
      if mode and a < 0:
        zero_point_candidtates.append((prev, i))
      elif not mode and a > 0:
        zero_point_candidtates.append((prev, i))
	
      prev = i
    
    # Calculate the perios
    zero_points = []
    
    for A1_idx, A2_idx in zero_point_candidtates:
      A1_idx, A2_idx = zero_point_candidtates[0][0], zero_point_candidtates[0][1]
      A1, A2 = abs(arr[A1_idx]), abs(arr[A2_idx])

      # Calculate the zero point
      zero_point = ( A1 / (A1 + A2) ) * sampling_period
      zero_points.append(zero_point)
      
    return 2* (zero_points[1] - zero_points[0])

# ---------------------------------------------------------------------------- #
#                            Island Leetcode Problem                           #
# ---------------------------------------------------------------------------- #
# cepton

# Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.
# An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
Example 2:
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3

Constraints:
m == grid.length
n == grid[i].length
1 <= m, n <= 300
grid[i][j] is '0' or '1'.


# Input variable is grid
def count_islands(grid):
  height, width = len(grid), len(grid[0])
  directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
  num_islands = 0

  def bfs(row, col):
    grid[row][col] = 0

    for oy, ox in directions:
      o_row, o_col = row + oy, col + o_x

      if 0 <= o_row < height and 0 <= o_col < width and grid[o_row][o_col] == 1:# within bounds and value in gris is == 1
        bfs(o_row, o_col)


  for row in range(height):
    for col in range(width):
      if grid[row][col] == 1:
        # BFS
        bfs(row, col)
        num_islands += 1
      else: # if 0
        pass
        
	return num_islands

grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
    
 
# ---------------------------------------------------------------------------- #
#                                   LRU Cache                                  #
# ---------------------------------------------------------------------------- #

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
2
​
3
Implement the LRUCache class:
4
​
5
LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
6
int get(int key) Return the value of the key if the key exists, otherwise return -1.
7
​
8
void put(int key, int value) Update the value of the key if the key exists. 
9
Otherwise, add the key-value pair to the cache. 
10
If the number of keys exceeds the capacity from this operation, evict the least recently used key.
11
​
12
The functions get and put must each run in O(1) average time complexity.
13
​
14
​
15
class LRUCache {
16
​
17
class Node:
18
​
19
public:
20
    LRUCache(int capacity) {
21
        # Dictionary (Key Val) {key, (value, idx_in_queue)}
22
        # Linked List (FIFO Queue) [List of keys]
23
        store = {}
24
        queue = deque() -> popleft()
25
        self.max_elem = capacity
26
    }
27
    
28
    int get(int key) {
29
        # inferring the dictionary
30
        if key in store:
31
          # Update the node position
32
          
33
          # Return 
34
          return store[key][0]
35
        else:
36
          return -1
37
    }
38
    
39
    void put(int key, int value) {
40
        # Check if capacity
41
          if len(store) == self.max_elem:
42
          
43
            # Pop first key from liked List .popleft()
44
            to_evicy = queue.popleft()
45
            
46
            # Delete that key from dictionary
47
            del(store[to_evict])
48
            
49
            # Create a new Node
50
            
51
            # Add key into dictionary
52
            store[key] = ()
53
            
54
            # Add key into Queue
55
          else:
56
            # Add the key into dictionary
57
            # .append()
58
          
59
          
60
    }
61
};
62
​
63
​



