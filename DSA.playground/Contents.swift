import Foundation
//
//class Solution {
//    func findDuplicate(_ nums: [Int]) -> Int {
//        /*
//        var slow = nums[0]
//        var fast = nums[0]
//        // Floyd Marshall Algorithm
//        // finding intersection point ( can be inside the circle also )
//        // initially both slow and fast are same, needs atleast one iteration here to happen for the while loop
//        repeat {
//            slow = nums[slow]
//            fast = nums[nums[fast]]
//        } while slow != fast
//        // finding the entrance point of the cycle
//        while slow != fast {
//            slow = nums[slow]
//            fast = nums[fast]
//        }
//        return slow
//         */
//        /*
//         Using Bit Manipulation
//         */
//        var n = nums.count
//        var result = 0
//        for bit in 0..<32 {
//            var count = 0
//            var expectedCount = 0
//            for num in nums {
//                if num & (1<<bit) == 1 {
//                    count += 1
//                }
//            }
//            for val in 1...n {
//                if val & (1<<bit) == 1 {
//                    expectedCount += 1
//                }
//            }
//            if count>expectedCount {
//                result = result | 1<<bit
//            }
//        }
//        return result
//    }
//}
//let solution = Solution()
//var nums1 = [4,0,0,0,0,0]
//var m = 1
//var nums2 = [1,2,3,5,6]
//var n = 5
////solution.merge(&nums1, m, nums2, n)
//let result = solution.findDuplicate([1,3,2,4,2])
//print(result)
//
//


/*
 4 3 2 6 1 2

 4 3 2      6 1 2

 3 4   2      1 6    2

 4   3   2      6  1    2

 inversionCount = 1, 2, 4,
 */

//let dictionary = ["Swapnil":10] // this is a dictionary [String:Int] does not take optional key here , key can't be optional
//let result = dictionary["Swapnil"] // result is an Int? optional because dictionary returns optional value
//print(result) // Optional(10)
//print(dictionary["Angela"])  // this return nil

//class A {
//    var a:Int
//    var b:Int
//    init(a: Int, b: Int) {
//        self.a = a
//        self.b = b
//    }
//}
//
//var a = A(a: 5, b: 10)
//print(a.a)

//let stringKey : String?
//stringKey = "Swapnil"
//if var unwrappedStringKey = stringKey {
//    print(unwrappedStringKey)
//}


//class Solution {
//    func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
//        // consider this as 1D array and find the element through binary search.
//        // for ith element in 1D array in 2D array it's represented as
//        // row = i/n , col = i%n
//        let m = matrix.count
//        let n = matrix[0].count
//        let start = 0
//        let end = m*n - 1
//        func binarySearch(_ start: Int, _ end: Int) -> Bool {
//            if start > end {
//                return false
//            }
//            let mid = (start + end)/2
//            let row = mid/n
//            let col = mid%n
//            if matrix[row][col] == target {
//                return true
//            } else if matrix[row][col] < target {
//                return binarySearch(mid+1, end)
//            } else {
//                return binarySearch(start, mid-1)
//            }
//        }
//        return binarySearch(start, end)
//    }
//}
//
//let solution = Solution()
//let matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
//print(solution.searchMatrix(matrix, 3))

class Solution {
    // Gives single majority element in the array i.e it's count is more than n/2
    //    func majorityElement(_ nums: [Int]) -> Int {
    //        var count = 1
    //        var candidate = nums[0]
    //        for i in 1..<nums.count {
    //            if nums[i] == candidate {
    //                count += 1
    //            } else {
    //                count -= 1
    //                if count == 0 {
    //                    candidate = nums[i]
    //                    count = 1
    //                }
    //            }
    //        }
    //        var sureCandidate = 0
    //        for num in nums {
    //            if num == candidate {
    //                sureCandidate += 1
    //            }
    //        }
    //        if sureCandidate > nums.count/2 {
    //            return candidate
    //        } else {
    //            return -1
    //        }
    //    }
    //    // Gives all majority elements in the array i.e their count is more than n/3
    //    func majorityElementII(_ nums: [Int]) -> [Int] {
    //        // Probable candidates are atmost 2, because if there are more than 2 elements then their count can't be more than n/3
    //        var count1 = 0, count2 = 0
    //        var candidate1 = -1, candidate2 = -1
    //        for j in 0..<nums.count {
    //            if nums[j] == candidate1 {
    //                count1 += 1
    //            } else if nums[j] == candidate2 {
    //                count2 += 1
    //            } else if count1 == 0 {
    //                candidate1 = nums[j]
    //                count1 = 1
    //            } else if count2 == 0 {
    //                candidate2 = nums[j]
    //                count2 = 1
    //            } else {
    //                count1 -= 1
    //                count2 -= 1
    //            }
    //        }
    //        // Now we have 2 candidates, we need to check their counts
    //        var ans = [Int]()
    //        var sureCandidateCount1 = 0, sureCandidateCount2 = 0
    //        for j in 0..<nums.count {
    //            if nums[j] == candidate1 {
    //                sureCandidateCount1 += 1
    //            } else if nums[j] == candidate2 {
    //                sureCandidateCount2 += 1
    //            }
    //        }
    //        if sureCandidateCount1 > nums.count/3 {
    //            ans.append(candidate1)
    //        }
    //        if sureCandidateCount2 > nums.count/3 && candidate2 != candidate1 {
    //            ans.append(candidate2)
    //        }
    //        return ans
    //    }

    //    func reversePairs(_ nums: [Int]) -> Int {
    ////        var answer = 0
    ////        for i in 0..<nums.count {
    ////            for j in i+1..<nums.count {
    ////                if nums[i] > 2*nums[j] {
    ////                    answer += 1
    ////                }
    ////            }
    ////        }
    ////        return answer
    //        if nums.count <= 1 { return 0 }
    //        var nums = nums
    //        return mergeSortCount(&nums, 0, nums.count - 1)
    //    }
    //
    //    func mergeSortCount(_ nums:inout [Int], _ low: Int, _ high: Int) -> Int {
    //        if low >= high { return 0 }
    //        let mid = low + (high - low)/2
    //        var count = 0
    //        count += mergeSortCount(&nums, low, mid)
    //        count += mergeSortCount(&nums, mid + 1, high)
    //        count += countReversePairsBetweenTwoSortedArrays(&nums, low, mid, high)
    //        mergeHelper(&nums, low, mid, high)
    //        return count
    //    }
    //
    //    func countReversePairsBetweenTwoSortedArrays(_ nums:inout [Int], _ low: Int,_ mid: Int, _ high: Int) -> Int {
    //        var leftArray = Array(nums[low...mid])
    //        var rightArray = Array(nums[mid+1...high])
    //        var j = 0
    //        var count = 0
    //        for i in 0..<leftArray.count {
    //            while j < rightArray.count {
    //                let doubleValue = 2.0 * Double(rightArray[j])
    //                if Double(leftArray[i]) > doubleValue {
    //                    j += 1
    //                } else {
    //                    break
    //                }
    //            }
    //            count += j
    //        }
    //        return count
    //    }
    //
    //    func mergeHelper(_ nums:inout [Int], _ low: Int,_ mid: Int, _ high: Int) {
    //        var leftArray = Array(nums[low...mid])
    //        var rightArray = Array(nums[mid+1...high])
    //        var leftStart = 0, rightStart = 0 , index = low
    //        while leftStart < leftArray.count && rightStart < rightArray.count {
    //            if leftArray[leftStart] < rightArray[rightStart] {
    //                nums[index] = leftArray[leftStart]
    //                leftStart += 1
    //            } else {
    //                nums[index] = rightArray[rightStart]
    //                rightStart += 1
    //            }
    //            index += 1
    //        }
    //        while leftStart < leftArray.count {
    //            nums[index] = leftArray[leftStart]
    //            leftStart += 1
    //            index += 1
    //        }
    //        while rightStart < rightArray.count {
    //            nums[index] = rightArray[rightStart]
    //            rightStart += 1
    //            index += 1
    //        }
    //    }
    //    func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
    //        var hashMap: [Int: Int] = [:]
    //        for (index, value) in nums.enumerated() {
    //            if let otherIndex = hashMap[target - value] {
    //                return [otherIndex, index]
    //            }
    //            hashMap[value] = index
    //        }
    //        return []
    //    }
    //    func fourSum(_ nums: [Int], _ target: Int) -> [[Int]] {
    //        let nums = nums.sorted()
    //        return kSum(nums, target, 0, 4)
    //    }
    //    func kSum (_ nums: [Int], _ target: Int,_ start: Int, _ k: Int) -> [[Int]] {
    //        var result = [[Int]]()
    //        let n = nums.count
    //        if k == 2{
    //            // 2 pointer approach, act as base case
    //            var left = start
    //            var right = n - 1
    //            while left < right {
    //                if nums[left] + nums[right] == target {
    //                    result.append([nums[left], nums[right]])
    //                    /*
    //                     Need to skip duplicates
    //                     */
    //                    while left<right && nums[left] == nums[left + 1] {
    //                        left += 1
    //                    }
    //                    while left<right && nums[right] == nums[right - 1] {
    //                        right -= 1
    //                    }
    //                    left += 1
    //                    right -= 1
    //                } else if nums[left] + nums[right] < target {
    //                    left += 1
    //                } else {
    //                    right -= 1
    //                }
    //            }
    //            return result
    //        }
    //        var i = start
    //        while i < n - k + 1 {
    //            if i>start {
    //                if nums[i] == nums[i-1]{
    //                    i += 1
    //                    continue
    //                }
    //            }
    //            let restOfArrayResult = kSum(nums, target - nums[i], i + 1, k - 1)
    //            for restOfArray in restOfArrayResult {
    //                result.append([nums[i]] + restOfArray)
    //            }
    //            i += 1
    //        }
    //        return result
    //    }
    //    func longestConsecutive(_ nums: [Int]) -> Int {
    ///*
    //        var numSet = Set(nums) // Using set for O(1) lookup and Set is an unordered collection
    //        var longestStreak = 0
    //        for num in numSet {
    //            if !numSet.contains(num-1){
    //                //Starting a new sequence
    //                var currentNum = num
    //                var currentStreak = 1
    //                while numSet.contains(currentNum + 1) {
    //                    currentStreak += 1
    //                    currentNum += 1
    //                }
    //                longestStreak = max(longestStreak, currentStreak)
    //            }
    //        }
    //        return longestStreak // This is O(n) solution, even though we are using 2 loops, the inner loop will cover the consecutive elements once and outer loop won't cover those elements again, so it is O(n) in total.
    // */
    //        // Using DSU to find longest consecutive sequence
    //        let uniqueNums = Array(Set(nums))
    //        let n = uniqueNums.count
    //        var uniqueNumsToIndex: [Int: Int] = [:]
    //        for (index, num) in uniqueNums.enumerated() {
    //            uniqueNumsToIndex[num] = index
    //        }
    //
    //        var dsu = DSU(n)
    //
    //        // Union or grouping the consecutive numbers
    //        for num in uniqueNums{
    //            if let nextNumIndex = uniqueNumsToIndex[num + 1] {
    //                dsu.union(uniqueNumsToIndex[num]!, nextNumIndex)
    //            }
    //        }
    //        return dsu.size.max() ?? 0 // Return the size of the largest group
    //    }
    /*
     Longest consecutive subsequence
     Kadane
     K sum
     */

    func subarraySum(_ nums: [Int], _ k: Int) -> Int {
        // needs to return the number of continuous subarrays whose sum equals to k
        /*
         // 1st approach is to use brute force, O(n^2) solution
         var prefixSum = Array(repeating: 0, count: nums.count)
         if nums.isEmpty {
         return 0
         }
         prefixSum[0] = nums[0]
         for i in 1..<nums.count {
         prefixSum[i] = prefixSum[i-1] + nums[i]
         }
         var count = 0
         for i in 0..<nums.count {
         for j in i..<nums.count {
         let subarray = Array(nums[i...j])  // just nums[i...j] will be a ArraySlice type and not an Array type
         let sum = subarray.reduce(0, +)
         if sum == k {
         count += 1

         }
         }
         }
         return count
         */
        /*
         2nd approach is to use a hashmap to store the prefix sum and its count
         */
        var count = 0
        var currentSum = 0
        var prefixSumCount: [Int: Int] = [0:1] // number of times a prefix sum has occurred
        for i in 0..<nums.count {
            currentSum += nums[i]
            let targetSumExists = prefixSumCount[currentSum-k] ?? 0 // Check if there exists a prefix sum such that currentSum - prefixSum = k
                count += targetSumExists
                prefixSumCount[currentSum, default: 0] += 1 // Initialize if not present
        }
        return count
    }

    func findLongestSubarrayWithSum(_ nums: [Int], _ k: Int) -> Int {
        // Returns the length of the longest subarray with sum equals to k
        var count = 0
        var currentSum = 0
        var prefixSumIndex: [Int: Int] = [0: -1] // Maps prefix sum to its first occurrence index
        for i  in 0..<nums.count {
            currentSum += nums[i]
            if let firstIndex = prefixSumIndex[currentSum - k] {
                count = max(count, i - firstIndex) // Update count if we found a longer subarray
            }
            // Only add the prefix sum if it is not already present to ensure we get the longest subarray
            if prefixSumIndex[currentSum] == nil {
                prefixSumIndex[currentSum] = i
            }
        }
        return count
    }

    func findSubarraysWithXorEqualsTok(_ nums: [Int], _ k: Int) -> Int {
        // if A XOR B = k, then A = k XOR B
        var count = 0
        var currentXor = 0
        var prefixXorCount: [Int: Int] = [0: 1] // Maps prefix XOR to its count
        for i in 0..<nums.count {
            currentXor ^= nums[i] // Update current XOR
            let targetXor = currentXor ^ k // Find the target XOR
            count += prefixXorCount[targetXor, default: 0] // Add the count of target XOR
            prefixXorCount[currentXor, default: 0] += 1 // Increment the count of current XOR
        }
        return count
    }

    func lengthOfLongestSubstring(_ s: String) -> Int {
        // example string: aecbcdeaf : Answer = 6 , i.e. bcdeaf
        // Idea have a hashMap which tells the index , i.e last occurrence of a character and use sliding window concept
        var maxLength = 0
        var start = 0
        var lastSeenCharacters: [Character: Int] = [:]
        for (index,character) in s.enumerated() {
            if let lastSeenIndex = lastSeenCharacters[character] {
                start = max(start, lastSeenIndex + 1)
            }
            maxLength = max(maxLength, index - start + 1)
            lastSeenCharacters[character] = index
        }
        return maxLength
    }

    func reverseList(_ head: ListNode?) -> ListNode? {
        /* 1st method iterative solution
        var prev: ListNode?
        var current = head
        while current != nil {
            let nextNode = current?.next
            current?.next = prev
            prev = current
            current = nextNode
        }
        return prev
         */
        //2nd method: Recursive solution
        if head == nil || head?.next == nil {
            return head
        }
        let newHead = reverseList(head?.next)
        head?.next?.next = head
        head?.next = nil

        return newHead
    }
    func middleNode(_ head: ListNode?) -> ListNode? {
        /* 1st method: simply finding the number of elements and traversing again to get that
        var sizeOfLinkedList = 0
        var currentNode: ListNode? = head
        while(currentNode?.next != nil){
            sizeOfLinkedList += 1
            currentNode = currentNode?.next
        }
        var currentNodeForTraversal: ListNode? = head
        for _ in 0..<(sizeOfLinkedList/2) {
            currentNodeForTraversal = currentNodeForTraversal?.next
        }
        return currentNodeForTraversal
         */
        // 2nd method: 2 pointer slow/fast approach
        var slow: ListNode? = head
        var fast: ListNode? = head
        while fast != nil && fast?.next != nil {
            slow = slow?.next
            fast = fast?.next?.next
        }
        return slow
    }
    /*
     3rd method: Recursive solution for finding middle of Linked List
     func findMiddleUtil(_ node: ListNode?, _ n: inout Int, _ mid: inout ListNode?) {
         guard let node = node else {
             n = (n+1) / 2   // Set n to half on reaching end
             return
         }
         n += 1
         findMiddleUtil(node.next, &n, &mid)
         n -= 1
         if n == 0 {
             mid = node
         }
     }

     func middleNode(_ head: ListNode?) -> ListNode? {
         var n = 0
         var mid: ListNode? = nil
         findMiddleUtil(head, &n, &mid)
         return mid
     }
     */
    func mergeTwoLists(_ list1: ListNode?, _ list2: ListNode?) -> ListNode? {
    let dummy = ListNode(0)
    var tail = dummy
    var node1 = list1
    var node2 = list2

    while let n1 = node1, let n2 = node2 {
        if n1.val < n2.val {
            tail.next = n1
            node1 = n1.next
        } else {
            tail.next = n2
            node2 = n2.next
        }
        tail = tail.next!
    }

    // One list may have leftovers
    tail.next = node1 ?? node2
    return dummy.next
    }

    func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
        // Approach 1: Finding the length of the list and then removing the nth node from the end
        // Approach 2: Using two pointers (fast and slow) to find the nth node from the end, keep the fast n steps ahead of the slow pointer, when fast reaches the end, slow will be at the nth node from the end
        var dummy = ListNode(0)
        dummy.next = head
        var fast: ListNode? = dummy
        var slow: ListNode? = dummy
        for _ in 0..<n {
            fast = fast?.next
        }
        while fast?.next != nil {
            slow = slow?.next
            fast = fast?.next
        }
        slow?.next = slow?.next?.next
        return dummy.next
    }

    func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        /*
         You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
         You may assume the two numbers do not contain any leading zero, except the number 0 itself.
         */
        // Need to use the concept of carry here
        var carry = 0
        let dummyNode = ListNode(0)
        var p = dummyNode
        var list1 = l1, list2 = l2
        while (list1 != nil || list2 != nil) {
            let val1 = list1?.val ?? 0
            let val2 = list2?.val ?? 0
            let sum = val1 + val2 + carry
            let newNodeVal = sum / 10
            carry = sum % 10
            p.next = ListNode(newNodeVal)
            if let next = p.next {
                p = next
            }
            list1 = list1?.next
            list2 = list2?.next
        }
        return dummyNode.next
    }

    func deleteNode(_ node: ListNode?) {
        var currentNode = node
        var nextNode = node?.next
        if currentNode != nil && nextNode != nil {
            currentNode?.val = nextNode?.val ?? 0
            currentNode?.next = nextNode?.next
        }
    }

    func getIntersectionNode(_ headA: ListNode?, _ headB: ListNode?) -> ListNode? {
        // find length of both the linked list
        var currentA = headA, currentB = headB
        var lengthA = 0, lengthB = 0
        while currentA != nil {
            lengthA += 1
            currentA = currentA?.next
        }
        while currentB != nil {
            lengthB += 1
            currentB = currentB?.next
        }
        // Move the pointer of the longer list by the difference in lengths
        var pointerToAdvance = lengthA > lengthB ? headA : headB
        for _ in 0..<abs(lengthA - lengthB) {
            pointerToAdvance = pointerToAdvance?.next
        }
        // Now move both pointers until they meet
        var pointerA = lengthA > lengthB ? pointerToAdvance : headA
        var pointerB = lengthA > lengthB ? headB : pointerToAdvance

        while pointerA !== pointerB {
            pointerA = pointerA?.next
            pointerB = pointerB?.next
        }
        return pointerA // or pointerB, both will be the same at this point
    }

    func hasCycle(_ head: ListNode?) -> Bool {
        // Method 1: Using a set to store visited nodes
        /*
        var visitedNodes: Set<ListNode> = []
        var currentNode = head
        while currentNode != nil {
            if visitedNodes.contains(currentNode!) {
                return true // Cycle detected
            }
            visitedNodes.insert(currentNode!)
            currentNode = currentNode?.next
        }
        return false
         */
        // Method 2: Using Floyd's Cycle Detection Algorithm (Tortoise and Hare)
        var dummyNode : ListNode? = ListNode(0)
        dummyNode?.next = head
        var slow = dummyNode
        var fast = dummyNode
        // here initially both slow and fast are same, so need to move them at least once, using do while loop
        repeat {
            slow = slow?.next
            fast = fast?.next?.next
            if fast == nil || fast?.next == nil {
                return false // No cycle
            }
        } while slow != fast
        // Meaningful cycle detected, move slow pointer to head and check if it meets fast pointer
        slow = dummyNode
        while slow !== fast {
            slow = slow?.next
            fast = fast?.next
            if fast == nil || fast?.next == nil {
                return false // No cycle
            }
        }
        return true // Cycle detected : slow will point that intersection point
    }

    func reverseKGroup(_ head: ListNode?, _ k: Int) -> ListNode? {
        // Intuition: Recursion
        var currentNode: ListNode? = head

        // Checking if k nodes do exist in Linked List
        for _ in 0..<k {
            if currentNode == nil {
                return head
            }
            currentNode = currentNode?.next
        }

        // Reverse the first k nodes
        var prev: ListNode? = nil
        var current = head
        var count = 0
        for _ in 0..<k {
            let nextNode = current?.next
            current?.next = prev
            prev = current
            current = nextNode
        }

        // head is the new tail, previous is the new head
        if let originalHead = head {
            originalHead.next = reverseKGroup(current, k)
        }

        return prev
    }

    func isPalindrome(_ head: ListNode?) -> Bool {
        // first find the middle of the linked list
        // reverse the second half of linked list
        // traverse if equal till the end : yes
        // restore the order of second half to original list
        if head == nil || head?.next == nil {
            return true
        }

        var middleNode = middleNode(head)
        // reverse the second half
        var secondHalfHead: ListNode? = reverseList(middleNode?.next)

        var firstHalfNode: ListNode? = head
        var isPalindrome = true

        while isPalindrome && secondHalfHead != nil {
            if firstHalfNode?.val != secondHalfHead?.val {
                isPalindrome = false
            }
            firstHalfNode = firstHalfNode?.next
            secondHalfHead = secondHalfHead?.next
        }

        // restore the second half again
        firstHalfNode?.next = reverseList(middleNode)
        return isPalindrome
    }
}

extension ListNode: Hashable {
    // Hashable conformance
    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))  // Hash based on memory address
    }

    public static func == (lhs: ListNode, rhs: ListNode) -> Bool {
        return lhs === rhs  // Check if same object (reference equality)
    }
}

public class ListNode {
    public var val: Int
    public var next: ListNode?
    public init() { self.val = 0; self.next = nil; }
    public init(_ val: Int) { self.val = val; self.next = nil; }
    public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
}

class DSU {
    var parent: [Int] // Parent of each node
    var size: [Int] // Size of each group
    init(_ parent: [Int], _ size: [Int]) {
        self.parent = parent
        self.size = size
    }

    init(_ n: Int) {
        self.parent = Array(0..<n) // Each node is its own parent initially
        self.size = Array(repeating: 1, count: n) // Each node is its own parent and size is 1 initially
    }

    func find(_ x: Int) -> Int {
        if parent[x] != x {
            parent[x] = find(parent[x])
        }
        return parent[x]
    }
    func union(_ x: Int, _ y: Int) {
        let parentOfX = find(x)
        let parentOfY = find(y)
        if parentOfX == parentOfY {
            return
        }
        if size[parentOfX] < size[parentOfY] {
            parent[parentOfX] = parentOfY
            size[parentOfY] += size[parentOfX]
        } else {
            parent[parentOfY] = parentOfX
            size[parentOfX] += size[parentOfY]
        }
    }
}

let solution = Solution()
//let nums = [1,0,-1,0,-2,2]
//let target = 0
//solution.fourSum(nums, target)

