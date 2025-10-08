import Foundation

//class Solution {
//
//    //Stacks and Queues: Part II
//    func maxOfMin(_ arr:[Int]) -> [Int] {
//        let n = arr.count
//
//        var previousSmallerIndexArray = getpreviousSmallerIndexArray(arr)
//        var nextSmallerIndexArrray = getnextSmallerIndexArray(arr)
//
//        print(previousSmallerIndexArray)
//        print(nextSmallerIndexArrray)
//
//        var output : [Int] = Array(repeating: Int.min, count: n)
//
//        for i in 0...n-1 {
//            let windowSize = nextSmallerIndexArrray[i] - previousSmallerIndexArray[i] - 1
//            output[windowSize-1] = max(output[windowSize-1], arr[i])
//        }
//
//        for i in stride(from: n-2, through: 0, by: -1){
//            output[i] = max(output[i], output[i+1])
//        }
//
//        return output
//    }
//
//    func getpreviousSmallerIndexArray(_ arr:[Int]) -> [Int] {
//        let n = arr.count
//        var output = Array(repeating: -1, count: n)
//        var stack : [Int] = [] //monotonic increasing stack containing the indices
//        stack.append(0)
//        for i in 1...n-1 {
//            while !stack.isEmpty && arr[stack.last!] >= arr[i] {
//                _ = stack.removeLast()
//            }
//            if !stack.isEmpty {
//                output[i] = stack.last!
//            }
//            stack.append(i)
//        }
//        return output
//    }
//
//
//    func getnextSmallerIndexArray(_ arr:[Int]) -> [Int] {
//        let n = arr.count
//        var output = Array(repeating: n, count: n)
//        var stack : [Int] = [] //monotonic increasing stack containing the indices
//        stack.append(n-1)
//        for i in stride(from: n-2, through: 0, by: -1) {
//            while !stack.isEmpty && arr[stack.last!] >= arr[i] {
//                _ = stack.removeLast()
//            }
//            if !stack.isEmpty {
//                output[i] = stack.last!
//            }
//            stack.append(i)
//        }
//        return output
//    }
//
//
//    /*
//     Case 1 : Find Celebrity :
//     If API is given knows(a,b) and we need to think about reducing this API also
//
//    func knows(_ a: Int, _ b: Int) -> Bool {
//        // Provided API
//        return true // or mocked
//    }
//
//    func findCelebrity(_ n: Int) -> Int {
//        /*
//         Using stack
//        var stack = Array(0..<n)
//
//        while stack.count > 1 {
//            let a = stack.removeLast()
//            let b = stack.removeLast()
//            if knows(a, b) {
//                stack.append(b)
//            } else {
//                stack.append(a)
//            }
//        }
//
//        let candidate = stack.last!
//        for i in 0..<n {
//            if i != candidate {
//                if knows(candidate, i) || !knows(i, candidate) {
//                    return -1
//                }
//            }
//        }
//        return candidate
//         */
//        // Using Two Pointer Approach
//        var left = 1, right = n
//
//        while left < right {
//            if knows(left,right) {
//                //left can't be the celebrity
//                left += 1
//            } else if knows(right,left) {
//                // right can't be the celebrity
//                right -= 1
//            } else {
//                left += 1
//                right -= 1
//            }
//        }
//
//        var possibleCandidate = left
//
//        for i in 1...n {
//            if i != possibleCandidate {
//                if knows(possibleCandidate,i) || !knows(i, possibleCandidate) {
//                    return -1
//                }
//            }
//        }
//
//        return possibleCandidate
//    }
//     */
//
//    /*
//     Case 2 Knows Matrix is given and then trying to find, the above approaches are all valid
//     Basically it would be that column with all 1s in knows matrix and row with all 0, starting from upper right corner
//     */
//    func findCelebrity(_ n: Int, _ knows: [[Int]]) -> Int {
//        var row = 0, col = n - 1
//
//        // Find potential candidate
//        while row < n && col >= 0 {
//            if knows[row][col] == 1 {
//                row += 1  // row person knows col, so row can't be celebrity
//            } else {
//                col -= 1  // row doesn't know col, so col can't be celebrity
//            }
//        }
//
//        // If we exit because col < 0, no valid candidate
//        if col < 0 { return -1 }
//
//        let candidate = col
//
//        // Verify candidate: check entire row and column
//        for i in 0..<n {
//            if i == candidate { continue }
//            if knows[candidate][i] == 1 || knows[i][candidate] == 0 {
//                return -1
//            }
//        }
//        return candidate
//    }
//}
//


class Solution {
    //    func findRedundantDirectedConnection(_ edges: [[Int]]) -> [Int] {
    //        /*
    //         Rooted Tree Properties:
    //         Every node except root has exactly one parent (indegree = 1)
    //
    //         Root has no parent (indegree = 0)
    //
    //         No cycles exist
    //
    //         Unique path from root to any node
    //
    //         Case Classification: You correctly identified that there are fundamentally two scenarios to handle.
    //
    //         What Needs Refinement 🔧
    //         Your case analysis is incomplete. Here's the more precise breakdown:
    //
    //         Case 1: A node has indegree > 1 (Two Parents)
    //         This happens when a node receives edges from multiple parents. This case can have two sub-scenarios:
    //
    //         Scenario 1a: Node with two parents + NO cycle
    //
    //         Solution: Remove the later occurring parent edge
    //
    //         Scenario 1b: Node with two parents + cycle exists
    //
    //         Solution: Remove the parent edge that participates in the cycle (not necessarily the later one)
    //
    //         Case 2: All nodes have indegree ≤ 1
    //         This means every node has at most one parent
    //
    //         Since we added exactly one extra edge, there must be a cycle
    //
    //         Solution: Remove any edge from the cycle (preferably the later occurring one)
    //
    //         The Key Insight You're Missing 🔍
    //         The critical distinction is: When there's a node with two parents, you need to determine if removing one of the parent edges would also break a potential cycle. The algorithm needs to:
    //
    //         First: Check if any node has indegree > 1
    //
    //         If yes: Identify the two candidate edges to remove
    //
    //         Then: Test which removal results in a valid rooted tree:
    //
    //         Try removing the second (later) parent edge first
    //
    //         If that creates a valid tree → that's your answer
    //
    //         If not → remove the first parent edge
    //
    //         Algorithm Framework:
    //         swift
    //         // Pseudo-code structure
    //         1. Build indegree array and parent tracking
    //         2. Find node with indegree > 1 (if exists)
    //         3. If such node exists:
    //         - Get two candidate edges
    //         - Try removing second candidate first
    //         - Check if resulting graph is valid rooted tree
    //         - If not, remove first candidate
    //         4. If no node has indegree > 1:
    //         - Detect cycle using DFS/Union-Find
    //         - Remove any edge from cycle (prefer later one)
    //         */
    //
    //        /*
    //         Case 1 : Example : [[1,2],[2,3],[3,4],[4,1],[1,5]] : Cycle exists
    //         Case 2 : Example : [[1,2],[1,3],[2,3]]
    //         Case 3 : Example : [[2,1],[3,1],[4,2],[1,4]]
    //         */
    //
    //        var candidate1 : [Int]? = nil //those edges which are causing a node to have 2 parents
    //        var candidate2 : [Int]? = nil
    //
    //        let n = edges.count
    //        var parent = Array(repeating: 0, count: n+1)
    //
    //        for edge in edges {
    //            let u = edge[0], v = edge[1]
    //            if parent[v] == 0 {
    //                parent[v] = u
    //            } else {
    //                candidate1 = [parent[v],v]
    //                candidate2 = edge
    //            }
    //        }
    //
    //        print(candidate1)
    //        print(candidate2)
    //
    //        //Building the adjacencyList
    //        var adjList : [Int: [Int]] = [:]
    //        for edge in edges {
    //            let u = edge[0], v = edge[1]
    //            if candidate2 != nil && candidate2 == [u,v] { continue } //Making the adjacency list without this potential candidate and then checking the conditions.
    //            adjList[u, default: []].append(v)
    //        }
    //
    //        print(adjList)
    //
    //        var visited = Array(repeating: 0 , count : n+1) //States 0: not visited, 1: visiting, 2 : visited
    //
    //        var cycleEdge : [Int]? = nil
    //
    //        //Tells if it contains cycle or not
    //        func dfs(_ node: Int) -> Bool {
    //            if adjList[node] == nil { return false }
    //            if visited[node] == 1 { return true }
    //            if visited[node] == 2 { return false }
    //            visited[node] = 1
    //            for neighbor in adjList[node]! {
    //                if dfs(neighbor) {
    //                    cycleEdge = [node, neighbor]
    //                    return true
    //                }
    //            }
    //            visited[node] = 2
    //            return false
    //        }
    //
    //        for i in 1...n {
    //            if visited[i] == 0 && dfs(i) { break }
    //        }
    //
    //        /*
    //         Case 1 : Example : [[1,2],[2,3],[3,4],[4,1],[1,5]] : Cycle exists
    //         Case 2 : Example : [[1,2],[1,3],[2,3]]
    //         Case 3 : Example : [[2,1],[3,1],[4,2],[1,4]]
    //         */
    //
    //        //No node with 2 parents : Case 1
    //        if candidate2 == nil {
    //            for edge in edges.reversed() {
    //                if visited[edge[0]] == 1 && visited[edge[1]] == 1 {
    //                    return edge //it expects you to return the last node of the cycle as per the input given : after DFS, traverse edges in reverse order and check which edge makes both endpoints "currently visited" (part of the cycle)
    //                }
    //            }
    //            return cycleEdge!
    //        }
    //
    //        // Case 2 : No cycle, removing candidate 2 will solve the issue
    //        if cycleEdge == nil {
    //            return candidate2!
    //        }
    //
    //        // Case 3: Cycle still exists, remove candidate1
    //        return candidate1!
    //    }
    /*
     Union Find Approach
     Intuition:
     Union Find (Disjoint Set Union–DSU) is ideal for cycle detection in graphs.
     
     Normally, for an undirected graph, you check whether adding an edge connects two vertices already in the same set (cycle).
     
     For Redundant Connection II (directed + possible two parents), you need to also record if any node gets two parents and handle both cycle and two-parent scenarios.
     
     Steps:
     
     Track parents of each node for the two-parent case. If a node has two different parent edges, record both candidates.
     
     Process edges by Union Find, skipping the second parent edge if a node with two parents is discovered.
     
     If a cycle occurs when adding an edge:
     
     If there’s a two-parent node, return the first parent edge (the earlier one).
     
     Else, return the edge that created the cycle (typically the last one in input).
     
     If all edges are processed with no cycles, the second parent edge is redundant.
     */
    
    func findRedundantDirectedConnection(_ edges: [[Int]]) -> [Int] {
        var candidate1 : [Int]? = nil //those edges which are causing a node to have 2 parents
        var candidate2 : [Int]? = nil
        
        let n = edges.count
        var parent = Array(repeating: 0, count: n+1)
        
        for edge in edges {
            let u = edge[0], v = edge[1]
            if parent[v] == 0 {
                parent[v] = u
            } else {
                candidate1 = [parent[v],v]
                candidate2 = edge
            }
        }
        
        var root = Array(0...n) //For Union Find
        
        //union find setup
        func unionFind(_ node: Int, _ root : inout [Int]) -> Int {
            if root[node] != node {
                root[node] = unionFind(root[node], &root)
            }
            return root[node]
        }
        
        for edge in edges {
            let u = edge[0], v = edge[1]
            if candidate2 != nil && candidate2 == [u,v] { continue }
            let ru = unionFind(u, &root)
            if ru == v {
                // Cycle exists
                if candidate1 != nil {
                    return candidate1!
                } else {
                    return edge
                }
            }
            root[v] = ru
        }
        // No cycles, second parent edge is the answer
        return candidate2!
    }
    
    
    func reverseWords(_ s: String) -> String {
        /*
         In Swift, String is not a fully mutable, in-place-editable structure due to its value semantics and copy-on-write behavior. You cannot truly reverse words “inplace” in a Swift String as you could in a mutable char array in C/C++ or Java, because Swift’s String is not just an array of characters—it manages Unicode scalars, grapheme clusters, and can reallocate memory internally upon mutation.
         
         So:
         
         O(N) time is achievable for reversing words in a string.
         
         O(1) extra space (in-place) is technically not achievable for Swift String due to its copy-on-write and immutability for string literals and function arguments. You can, however, approach O(1) space with [String] (character array) if constraints are relaxed, but it's not truly in-place at the byte level.
         
         In summary: Swift String is not mutable in the C/C++ sense, and you cannot do true in-place word reversal with O(1) space. Some auxiliary space or character array conversion is necessary.
         */
        
        //        var trimmedString = s.trimmingCharacters(in: .lowercaseLetters)  // "   hell world  " -> "hell world"
        //        let words = trimmedString.components(separatedBy: " ") // ["hell", "world"]
        //        let reversedWords = words.reversed() // ["world", "hell"]
        //        let answerString = String(reversedWords.joined(separator: " ")) //world hell
        
        var stringArrayWithNoExtraWhiteSpace = s.split(separator: " ") // "   hell   world  " -> ["hell", "world"]
        var reversedWordArray = stringArrayWithNoExtraWhiteSpace.reversed() // ["world", "hell"]
        var answerString = reversedWordArray.joined(separator: " ") // "world hell"
        return answerString
    }
    
    func longestPalindrome(_ s: String) -> String {
        var strArray = Array(s)
        let n = strArray.count
        var dp = Array(repeating: Array(repeating: false, count:n), count: n)
        var maxLen = 1
        var start = 0
        
        for len in 1...n {
            for i in 0...n-len {
                var j = i + len - 1
                if len == 1 {
                    dp[i][j] = true
                } else if len == 2 {
                    dp[i][j] = (strArray[i] == strArray[j])
                } else {
                    dp[i][j] = dp[i+1][j-1] && (strArray[i] == strArray[j])
                }
                if dp[i][j] && maxLen < len {
                    start = i
                    maxLen = len
                }
            }
        }
        return String(strArray[start..<start+maxLen])
    }
    func isAnagram(_ s: String, _ t: String) -> Bool {
        /*
         Got it, so you're starting with the optimization to directly return `false` if `s` and `t` have different lengths—that's a good preliminary check.
         
         Now, as for **Unicode**, let's dive into it:
         Unicode is a universal character encoding standard that covers almost all the characters and symbols used in the world's writing systems. Instead of just lowercase English letters (`a-z`), Unicode includes characters like `á`, `ü`, `漢`, `😊`, etc.
         
         Here's a question to test your understanding:
         If we need to handle Unicode characters in this problem, how would your approach change? Would the fixed-size array (of length 26) still be viable?
         */
        
        //Swift Character type is already a Unicode-compliant
        //        var frequencyHashMapForS : [Character:Int] = [:]
        //        var frequencyHashMapForT : [Character:Int] = [:]
        //
        //        if s.count != t.count {
        //            return false
        //        }
        //
        //        var sArray = Array(s)
        //        var tArray = Array(t)
        //
        //        let n = sArray.count
        //
        //        for i in 0...n-1 {
        //            frequencyHashMapForS[sArray[i],default: 0] += 1
        //            frequencyHashMapForT[tArray[i],default: 0] += 1
        //        }
        //
        //        return frequencyHashMapForS == frequencyHashMapForT
        if s.count != t.count {
            return false
        }
        
        var frequencyHashMap: [Character: Int] = [:]
        
        for char in s {
            frequencyHashMap[char, default: 0] += 1
        }
        
        for char in t {
            frequencyHashMap[char, default: 0] -= 1
        }
        
        // Check if all values in the hash map are 0
        for (_, value) in frequencyHashMap {
            if value != 0 {
                return false
            }
        }
        
        return true
    }
    
    
    func mergeKLists(_ lists: [ListNode?]) -> ListNode? {
        var head : ListNode? = nil
        
        if lists.count == 1 {
            return lists.first!
        }
        
        var minHeap = MinHeap()
        
        //insert all the head of lists
        for list in lists {
            minHeap.insert(list)
        }
        
        print(minHeap)
        
        head = minHeap.extractMin()
        
        var currNode = head
        
        while !minHeap.isEmpty {
            
            minHeap.insert(currNode?.next)
            
            var currMinNode = minHeap.extractMin()
            
            currNode?.next = currMinNode
            
            currNode = currMinNode
        }
        
        return head
    }
    
}

class MinHeap {
    private var minHeap : [ListNode] = []

    var peek : ListNode? {
        return minHeap.first
    }

    var isEmpty : Bool {
        return minHeap.isEmpty
    }

    var count : Int {
        return minHeap.count
    }
    
    //heapify up
    func insert(_ node: ListNode?) {
        guard let node = node else { return }
        
        minHeap.append(node)
        var childIdx = count - 1
        
        while true {
            var parentIdx = (childIdx - 1)/2
            if minHeap[parentIdx].val < minHeap[childIdx].val {
                break
            }
            minHeap.swapAt(parentIdx, childIdx)
            childIdx = parentIdx
        }
    }
    
    //Heapify Down
    func extractMin() -> ListNode? {
        if isEmpty {
            return nil
        }
        
        let answer = minHeap.first!
        minHeap[0] = minHeap.removeLast()
        
        var parentIdx = 0
        
        while parentIdx < count {
            let leftChildIdx = 2 * parentIdx + 1
            let rightChildIdx = 2 * parentIdx + 2
            
            var smallestIdx = parentIdx
            
            if leftChildIdx < count && minHeap[smallestIdx].val > minHeap[leftChildIdx].val {
                smallestIdx = leftChildIdx
            }
            
            if rightChildIdx < count && minHeap[smallestIdx].val > minHeap[rightChildIdx].val {
                smallestIdx = rightChildIdx
            }
            
            if smallestIdx == parentIdx {
                break
            }
            
            minHeap.swapAt(parentIdx, smallestIdx)
            parentIdx = smallestIdx
        }
        
        return answer
    }
}

 public class ListNode {
      public var val: Int
      public var next: ListNode?
      public init() { self.val = 0; self.next = nil; }
      public init(_ val: Int) { self.val = val; self.next = nil; }
      public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
}
 

let solution = Solution()

//print(solution.maxOfMin([10,20,30,50,10,70,30]))
//print(solution.findRedundantDirectedConnection([[1,2],[2,3],[3,4],[4,1],[1,5]]))
//print(solution.reverseWords("  hell   world "))
//print(solution.longestPalindrome("cbbd"))
//print(solution.isAnagram("😂😍SwapnilDhiman", "😂Dhiman😍Swapnil"))

var node1 = ListNode(10)
var node2 = ListNode(20)
var node3 = ListNode(30)

node1.next = node2
node2.next = node3

var node4 = ListNode(12)
var node5 = ListNode(22)

node4.next = node5

//print(solution.mergeKLists([node1,node4]))
print("Swapnil Dhiman")
