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
        
        while childIdx > 0 {
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
        
        if count == 1 {
            return minHeap.removeLast()
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

class TrieNode {
    var children : [Character:TrieNode] = [:]
}

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
        
        if lists.count == 1 {
            return lists.first!
        }
        
        var minHeap = MinHeap()
        
        //insert all the head of lists
        for list in lists {
            minHeap.insert(list)
        }
        
        let dummyHead = ListNode(0)
        var currNode = dummyHead
        
        while !minHeap.isEmpty {
            
            guard let node = minHeap.extractMin() else { break }
            
            currNode.next = node
            currNode = node
            
            if let nextNode = node.next {
                minHeap.insert(nextNode)
            }
        }
        
        return dummyHead.next
    }
    
    func myAtoi(_ s: String) -> Int {
        /*
         The algorithm for myAtoi(string s) is as follows:
         
         Whitespace: Ignore any leading whitespace (" ").
         Signedness: Determine the sign by checking if the next character is '-' or '+', assuming positivity if neither present.
         Conversion: Read the integer by skipping leading zeros until a non-digit character is encountered or the end of the string is reached. If no digits were read, then the result is 0.
         
         Rounding: If the integer is out of the 32-bit signed integer range [-231, 231 - 1], then round the integer to remain in the range. Specifically, integers less than -231 should be rounded to -231, and integers greater than 231 - 1 should be rounded to 231 - 1.
         */
        
        /*
         1. remove leading whitespaces
         2. if -, + , 0-9 character not encountered at prefix return 0
         3. +,- keep that in signed variable,default = +
         4. remove leading 0s if any present
         5. Int conversion till we are encountering 0-9
         6. Add - if signed variable has - sign
         */
        
        
        // removing whitespaces first
        let stringWithoutWhiteSpaces = s.trimmingCharacters(in: .whitespaces)
        if stringWithoutWhiteSpaces.isEmpty { return 0 }
        
        var signed : Character = "+"
        
        var stringArray = Array(stringWithoutWhiteSpaces)
        
        if stringArray[0] == "-" {
            signed = "-"
            stringArray.remove(at: 0)
        } else if stringArray[0] == "+" {
            stringArray.remove(at: 0)
        } else if !(stringArray[0].isNumber) {
            return 0
        }
        
        // removing any leading 0s
        var stringArrayWithoutLeading0s = stringArray.trimmingPrefix(while: { $0 == "0" })
        if stringArrayWithoutLeading0s.isEmpty { return 0 }
        
        var number : Int = 0
        for char in stringArrayWithoutLeading0s {
            if char.isNumber {
                let digit = Int(String(char))!
                // check for overflow and underflow
                if signed == "-" {
                    if -number < (Int(Int32.min) + digit) / 10 {
                        return Int(Int32.min)
                    }
                } else {
                    if number > (Int(Int32.max) - digit) / 10 {
                        return Int(Int32.max)
                    }
                }
                number = number * 10 + digit
                /*
                 
                 ## The Mathematical Logic
                 
                 The overflow check works by **reversing the operation** and checking if we're already at the dangerous boundary.
                 
                 ### For Positive Numbers (Int32.max = 2,147,483,647)
                 
                 **The Problem:**
                 - We want to do: `number = number * 10 + digit`
                 - But if this would exceed `Int32.max`, we need to catch it beforehand
                 
                 **The Solution - Reverse Engineering:**
                 Instead of checking if `number * 10 + digit > Int32.max`, we rearrange the math:
                 
                 ```
                 number * 10 + digit > Int32.max
                 number > (Int32.max - digit) / 10
                 ```
                 
                 So we check: **"Is the current number already too big that adding one more digit would overflow?"**
                 
                 ### Concrete Example:
                 
                 Let's say `Int32.max = 2,147,483,647` and we have:
                 - Current `number = 214,748,364`
                 - Next `digit = 9`
                 
                 **Step 1:** Calculate the threshold
                 ```
                 threshold = (Int32.max - digit) / 10
                 threshold = (2,147,483,647 - 9) / 10
                 threshold = 2,147,483,638 / 10
                 threshold = 214,748,363
                 ```
                 
                 **Step 2:** Check if we're past the threshold
                 ```
                 number > threshold?
                 214,748,364 > 214,748,363?
                 YES! We're already too big!
                 ```
                 
                 **Step 3:** What would happen if we proceeded?
                 ```
                 number * 10 + digit = 214,748,364 * 10 + 9 = 2,147,483,649
                 ```
                 This exceeds `Int32.max` (2,147,483,647), so we'd get overflow!
                 
                 ### For Negative Numbers (Int32.min = -2,147,483,648)
                 
                 The logic is similar but for the lower bound:
                 
                 **The Problem:**
                 - We want to do: `number = number * 10 + digit`, then negate it
                 - But if `-number` would go below `Int32.min`, we need to catch it
                 
                 **The Solution:**
                 ```
                 -(number * 10 + digit) < Int32.min
                 -(number * 10) - digit < Int32.min
                 -number * 10 < Int32.min + digit
                 -number < (Int32.min + digit) / 10
                 ```
                 
                 ### Why This Works:
                 
                 1. **We check BEFORE doing the operation** - not after
                 2. **We use integer division** - which naturally rounds down, making our threshold slightly more conservative
                 3. **We're checking the "last safe value"** before adding the next digit would cause overflow
                 
                 ### Visual Example:
                 
                 ```
                 Int32.max = 2147483647
                 
                 Safe values: 0, 1, 2, ..., 214748363
                 Dangerous:   214748364, 214748365, ...
                 ```
                 
                 When `number = 214748364`, we check:
                 - `214748364 > (2147483647 - 9) / 10`
                 - `214748364 > 214748363`
                 - `true` → We're in the danger zone!
                 
                 The key insight is that we're not trying to detect overflow after it happens - we're **predicting** it by checking if we're already at the boundary where the next operation would be unsafe.
                 
                 This is a common pattern in overflow-safe programming: **check the preconditions rather than trying to handle the overflow after it occurs**.
                 
                 */
            } else {
                break
            }
        }
        return signed == "-" ? -number : number
    }
    
    func longestCommonPrefix(_ strs: [String]) -> String {
        
        //        let smallestString = strs.min { a, b in
        //            a.count < b.count
        //        }
        //
        //        var answer = smallestString!
        //
        //        for str in strs {
        //            var commonPrefixString = str.commonPrefix(with: answer)
        //            if commonPrefixString == "" {
        //                return ""
        //            }
        //            if commonPrefixString.count < answer.count {
        //                answer = commonPrefixString
        //            }
        //        }
        //
        //        return answer
        
        var root = TrieNode()
        
        if strs.count == 1 {
            return strs.first!
        }
        
        let smallestString = strs.min { a, b in
            a.count < b.count
        }
        
        let smallestStringCount = smallestString!.count
        
        //Building the Trie
        for str in strs {
            var node = root
            var currStringArray = Array(str)
            for i in 0..<smallestStringCount {
                if node.children[currStringArray[i]] == nil {
                    node.children[currStringArray[i]] = TrieNode()
                }
                node = node.children[currStringArray[i]]!
            }
        }
        
        var answer = ""
        
        var node : TrieNode? = root
        
        //finding the first divergence
        while let currNode = node {
            let children = currNode.children
            if children.count > 1 || children.isEmpty {
                break
            }
            for (ch,nextChild) in children {
                node = nextChild
                answer += String(ch)
                break
            }
        }
        
        return answer
    }
    
    func repeatedStringMatch(_ a: String, _ b: String) -> Int {
        var maxRepeatedTimes = b.count/a.count
        var count = 0
        var a = a
        while count < maxRepeatedTimes {
            if kmpStringMatch(a,b) == -1 {
                count += 1
                a += a
            } else {
                break
            }
        }
        return count
    }
    
    func kmpStringMatch(_ s: String, _ t: String) -> Int {
        let lpsOfT = createLPS(t)
        var tPointer = 0, sPointer = 0
        
        var sArray = Array(s)
        var tArray = Array(t)
        
        var start = 0
        
        while tPointer < tArray.count && sPointer < sArray.count {
            if tArray[tPointer] == sArray[sPointer] {
                tPointer += 1
                sPointer += 1
            } else {
                if tPointer == 0 {
                    sPointer += 1 //Both are not getting matched
                } else {
                    tPointer = lpsOfT[tPointer - 1]
                }
            }
        }
        
        return tPointer == tArray.count ? sPointer - tPointer : -1
    }
    
    func createLPS(_ s: String) -> [Int] {
        // Create Longest Prefix Suffix ( longest prefix which is also a suffix for that index )
        var lps = Array(repeating:0, count: s.count)
        var prefix = 0 , suffix = 1
        var strArray = Array(s)
        while suffix < s.count {
            if strArray[prefix] == strArray[suffix] {
                lps[suffix] = prefix + 1
                prefix += 1
                suffix += 1
            } else {
                if prefix == 0 {
                    lps[suffix] = 0
                    suffix += 1
                } else {
                    prefix = lps[prefix - 1]
                }
            }
        }
        return lps
    }
    
    func fourSum(_ nums: [Int], _ target: Int) -> [[Int]] {
        let sortedNums = nums.sorted()
        return kSum(sortedNums,0,4,target)
    }
    
    func kSum(_ nums: [Int], _ start: Int, _ k : Int, _ target : Int) -> [[Int]] {
        let n = nums.count
        var results = [[Int]]()
        
        
        if k == 2 {
            var left = start
            var right = n - 1
            while left < right {
                if nums[left] + nums[right] == target {
                    results.append([nums[left], nums[right]])
                    //Handling duplicates
                    while left < right && nums[left] == nums[left-1] {
                        left += 1
                    }
                    while right > left && nums[right] == nums[right+1] {
                        right -= 1
                    }
                    left += 1
                    right -= 1
                }
                else if nums[left] + nums[right] < target {
                    left += 1
                } else {
                    right -= 1
                }
            }
            return results
        }
        
        var index = start
        
        while index < n - k + 1 {
            if index > start {
                if nums[index] == nums[index-1] {
                    index += 1
                    continue
                }
            }
            
            let subArraysConsideringCurrElement = kSum(nums, index+1, k-1, target - nums[index])
            for subArray in subArraysConsideringCurrElement {
                results.append([nums[index]] + subArray)
            }
            
            index += 1
        }
        
        return results
    }
    
    func minInsertions(_ s: String) -> Int {
        /*
         This approach of thinking that since palindrome is a string where the first half is the mirror image of second half, thus all the characters except one ( middle frequency character ) should have even frequency, thus finding the odd frequency character count is false ,
         
         Example: "aabcbc" ( all characters have even frequency and still not palindrome), "ccewnxhytsr"
         
         if s.count == 1 {
         return 0
         }
         var frequencyCharacter : [Character:Int] = [:]
         var strArray = Array(s)
         for ch in strArray {
         frequencyCharacter[ch, default: 0] += 1
         }
         
         var oddFrequencyCounts = 0
         
         for (ch,value) in frequencyCharacter {
         if value % 2 != 0 {
         oddFrequencyCounts += 1
         }
         }
         
         var middleCharacter = strArray[(strArray.count)/2]
         
         print(middleCharacter)
         
         return frequencyCharacter[middleCharacter]! % 2 == 0 ? oddFrequencyCounts : oddFrequencyCounts - 1
         }
         */
        
        /*
         Approach 2 :
         If we know the longest palindromic sub-sequence is x and the length of the string is n then, what is the answer to this problem? It is n - x as we need n - x insertions to make the remaining characters also palindrome.
         */
        
        //dp[i] : Length of longest palindromic substring ending at i th index
        
        var strArray = Array(s)
        var dp = Array(repeating: 1, count: strArray.count) //since every single character is a palindrome of length
        
        for i in 1..<strArray.count {
            let rangeStart = i-1-dp[i-1]
            if rangeStart >= 0 {
                if strArray[rangeStart] == strArray[i] {
                    dp[i] = dp[i-1] + 2
                }
            }
        }
        
        let longestPalindromicSubstringCount = dp.max()
        
        return strArray.count - longestPalindromicSubstringCount!
    }
    
}

let solution = Solution()
//print(solution.myAtoi("0-1"))
//print(solution.longestCommonPrefix(["flower","flow","flight"]))
//print(solution.maxOfMin([10,20,30,50,10,70,30]))
//print(solution.findRedundantDirectedConnection([[1,2],[2,3],[3,4],[4,1],[1,5]]))
//print(solution.reverseWords("  hell   world "))
//print(solution.longestPalindrome("cbbd"))
//print(solution.isAnagram("😂😍SwapnilDhiman", "😂Dhiman😍Swapnil"))
//
//var node1 = ListNode(10)
//var node2 = ListNode(20)
//var node3 = ListNode(30)
//
//node1.next = node2
//node2.next = node3
//
//var node4 = ListNode(12)
//var node5 = ListNode(22)
//
//node4.next = node5
//
//solution.mergeKLists([node1,node4])

//print(solution.repeatedStringMatch("abcd","cdabcdab"))

//print(solution.createLPS("aabaabaaa"))

//print(solution.repeatedStringMatch("abcd", "cdabcdab"))
//print(solution.fourSum([1,0,-1,0,-2,2], 0))
//print(solution.createLPS("cbcbaa"))
print(solution.minInsertions("zzazz"))
