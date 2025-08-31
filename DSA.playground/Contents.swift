import Foundation
//////////import Foundation
////////////
////////////class Solution {
////////////    func findDuplicate(_ nums: [Int]) -> Int {
////////////        /*
////////////        var slow = nums[0]
////////////        var fast = nums[0]
////////////        // Floyd Marshall Algorithm
////////////        // finding intersection point ( can be inside the circle also )
////////////        // initially both slow and fast are same, needs atleast one iteration here to happen for the while loop
////////////        repeat {
////////////            slow = nums[slow]
////////////            fast = nums[nums[fast]]
////////////        } while slow != fast
////////////        // finding the entrance point of the cycle
////////////        while slow != fast {
////////////            slow = nums[slow]
////////////            fast = nums[fast]
////////////        }
////////////        return slow
////////////         */
////////////        /*
////////////         Using Bit Manipulation
////////////         */
////////////        var n = nums.count
////////////        var result = 0
////////////        for bit in 0..<32 {
////////////            var count = 0
////////////            var expectedCount = 0
////////////            for num in nums {
////////////                if num & (1<<bit) == 1 {
////////////                    count += 1
////////////                }
////////////            }
////////////            for val in 1...n {
////////////                if val & (1<<bit) == 1 {
////////////                    expectedCount += 1
////////////                }
////////////            }
////////////            if count>expectedCount {
////////////                result = result | 1<<bit
////////////            }
////////////        }
////////////        return result
////////////    }
////////////}
////////////let solution = Solution()
////////////var nums1 = [4,0,0,0,0,0]
////////////var m = 1
////////////var nums2 = [1,2,3,5,6]
////////////var n = 5
//////////////solution.merge(&nums1, m, nums2, n)
////////////let result = solution.findDuplicate([1,3,2,4,2])
////////////print(result)
////////////
////////////
//////////
//////////
///////////*
////////// 4 3 2 6 1 2
//////////
////////// 4 3 2      6 1 2
//////////
////////// 3 4   2      1 6    2
//////////
////////// 4   3   2      6  1    2
//////////
////////// inversionCount = 1, 2, 4,
////////// */
//////////
////////////let dictionary = ["Swapnil":10] // this is a dictionary [String:Int] does not take optional key here , key can't be optional
////////////let result = dictionary["Swapnil"] // result is an Int? optional because dictionary returns optional value
////////////print(result) // Optional(10)
////////////print(dictionary["Angela"])  // this return nil
//////////
////////////class A {
////////////    var a:Int
////////////    var b:Int
////////////    init(a: Int, b: Int) {
////////////        self.a = a
////////////        self.b = b
////////////    }
////////////}
////////////
////////////var a = A(a: 5, b: 10)
////////////print(a.a)
//////////
////////////let stringKey : String?
////////////stringKey = "Swapnil"
////////////if var unwrappedStringKey = stringKey {
////////////    print(unwrappedStringKey)
////////////}
//////////
//////////
////////////class Solution {
////////////    func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
////////////        // consider this as 1D array and find the element through binary search.
////////////        // for ith element in 1D array in 2D array it's represented as
////////////        // row = i/n , col = i%n
////////////        let m = matrix.count
////////////        let n = matrix[0].count
////////////        let start = 0
////////////        let end = m*n - 1
////////////        func binarySearch(_ start: Int, _ end: Int) -> Bool {
////////////            if start > end {
////////////                return false
////////////            }
////////////            let mid = (start + end)/2
////////////            let row = mid/n
////////////            let col = mid%n
////////////            if matrix[row][col] == target {
////////////                return true
////////////            } else if matrix[row][col] < target {
////////////                return binarySearch(mid+1, end)
////////////            } else {
////////////                return binarySearch(start, mid-1)
////////////            }
////////////        }
////////////        return binarySearch(start, end)
////////////    }
////////////}
////////////
////////////let solution = Solution()
////////////let matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
////////////print(solution.searchMatrix(matrix, 3))
//////////
//////////class Solution {
//////////    // Gives single majority element in the array i.e it's count is more than n/2
//////////    //    func majorityElement(_ nums: [Int]) -> Int {
//////////    //        var count = 1
//////////    //        var candidate = nums[0]
//////////    //        for i in 1..<nums.count {
//////////    //            if nums[i] == candidate {
//////////    //                count += 1
//////////    //            } else {
//////////    //                count -= 1
//////////    //                if count == 0 {
//////////    //                    candidate = nums[i]
//////////    //                    count = 1
//////////    //                }
//////////    //            }
//////////    //        }
//////////    //        var sureCandidate = 0
//////////    //        for num in nums {
//////////    //            if num == candidate {
//////////    //                sureCandidate += 1
//////////    //            }
//////////    //        }
//////////    //        if sureCandidate > nums.count/2 {
//////////    //            return candidate
//////////    //        } else {
//////////    //            return -1
//////////    //        }
//////////    //    }
//////////    //    // Gives all majority elements in the array i.e their count is more than n/3
//////////    //    func majorityElementII(_ nums: [Int]) -> [Int] {
//////////    //        // Probable candidates are atmost 2, because if there are more than 2 elements then their count can't be more than n/3
//////////    //        var count1 = 0, count2 = 0
//////////    //        var candidate1 = -1, candidate2 = -1
//////////    //        for j in 0..<nums.count {
//////////    //            if nums[j] == candidate1 {
//////////    //                count1 += 1
//////////    //            } else if nums[j] == candidate2 {
//////////    //                count2 += 1
//////////    //            } else if count1 == 0 {
//////////    //                candidate1 = nums[j]
//////////    //                count1 = 1
//////////    //            } else if count2 == 0 {
//////////    //                candidate2 = nums[j]
//////////    //                count2 = 1
//////////    //            } else {
//////////    //                count1 -= 1
//////////    //                count2 -= 1
//////////    //            }
//////////    //        }
//////////    //        // Now we have 2 candidates, we need to check their counts
//////////    //        var ans = [Int]()
//////////    //        var sureCandidateCount1 = 0, sureCandidateCount2 = 0
//////////    //        for j in 0..<nums.count {
//////////    //            if nums[j] == candidate1 {
//////////    //                sureCandidateCount1 += 1
//////////    //            } else if nums[j] == candidate2 {
//////////    //                sureCandidateCount2 += 1
//////////    //            }
//////////    //        }
//////////    //        if sureCandidateCount1 > nums.count/3 {
//////////    //            ans.append(candidate1)
//////////    //        }
//////////    //        if sureCandidateCount2 > nums.count/3 && candidate2 != candidate1 {
//////////    //            ans.append(candidate2)
//////////    //        }
//////////    //        return ans
//////////    //    }
//////////
//////////    //    func reversePairs(_ nums: [Int]) -> Int {
//////////    ////        var answer = 0
//////////    ////        for i in 0..<nums.count {
//////////    ////            for j in i+1..<nums.count {
//////////    ////                if nums[i] > 2*nums[j] {
//////////    ////                    answer += 1
//////////    ////                }
//////////    ////            }
//////////    ////        }
//////////    ////        return answer
//////////    //        if nums.count <= 1 { return 0 }
//////////    //        var nums = nums
//////////    //        return mergeSortCount(&nums, 0, nums.count - 1)
//////////    //    }
//////////    //
//////////    //    func mergeSortCount(_ nums:inout [Int], _ low: Int, _ high: Int) -> Int {
//////////    //        if low >= high { return 0 }
//////////    //        let mid = low + (high - low)/2
//////////    //        var count = 0
//////////    //        count += mergeSortCount(&nums, low, mid)
//////////    //        count += mergeSortCount(&nums, mid + 1, high)
//////////    //        count += countReversePairsBetweenTwoSortedArrays(&nums, low, mid, high)
//////////    //        mergeHelper(&nums, low, mid, high)
//////////    //        return count
//////////    //    }
//////////    //
//////////    //    func countReversePairsBetweenTwoSortedArrays(_ nums:inout [Int], _ low: Int,_ mid: Int, _ high: Int) -> Int {
//////////    //        var leftArray = Array(nums[low...mid])
//////////    //        var rightArray = Array(nums[mid+1...high])
//////////    //        var j = 0
//////////    //        var count = 0
//////////    //        for i in 0..<leftArray.count {
//////////    //            while j < rightArray.count {
//////////    //                let doubleValue = 2.0 * Double(rightArray[j])
//////////    //                if Double(leftArray[i]) > doubleValue {
//////////    //                    j += 1
//////////    //                } else {
//////////    //                    break
//////////    //                }
//////////    //            }
//////////    //            count += j
//////////    //        }
//////////    //        return count
//////////    //    }
//////////    //
//////////    //    func mergeHelper(_ nums:inout [Int], _ low: Int,_ mid: Int, _ high: Int) {
//////////    //        var leftArray = Array(nums[low...mid])
//////////    //        var rightArray = Array(nums[mid+1...high])
//////////    //        var leftStart = 0, rightStart = 0 , index = low
//////////    //        while leftStart < leftArray.count && rightStart < rightArray.count {
//////////    //            if leftArray[leftStart] < rightArray[rightStart] {
//////////    //                nums[index] = leftArray[leftStart]
//////////    //                leftStart += 1
//////////    //            } else {
//////////    //                nums[index] = rightArray[rightStart]
//////////    //                rightStart += 1
//////////    //            }
//////////    //            index += 1
//////////    //        }
//////////    //        while leftStart < leftArray.count {
//////////    //            nums[index] = leftArray[leftStart]
//////////    //            leftStart += 1
//////////    //            index += 1
//////////    //        }
//////////    //        while rightStart < rightArray.count {
//////////    //            nums[index] = rightArray[rightStart]
//////////    //            rightStart += 1
//////////    //            index += 1
//////////    //        }
//////////    //    }
//////////    //    func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
//////////    //        var hashMap: [Int: Int] = [:]
//////////    //        for (index, value) in nums.enumerated() {
//////////    //            if let otherIndex = hashMap[target - value] {
//////////    //                return [otherIndex, index]
//////////    //            }
//////////    //            hashMap[value] = index
//////////    //        }
//////////    //        return []
//////////    //    }
//////////    //    func fourSum(_ nums: [Int], _ target: Int) -> [[Int]] {
//////////    //        let nums = nums.sorted()
//////////    //        return kSum(nums, target, 0, 4)
//////////    //    }
//////////    //    func kSum (_ nums: [Int], _ target: Int,_ start: Int, _ k: Int) -> [[Int]] {
//////////    //        var result = [[Int]]()
//////////    //        let n = nums.count
//////////    //        if k == 2{
//////////    //            // 2 pointer approach, act as base case
//////////    //            var left = start
//////////    //            var right = n - 1
//////////    //            while left < right {
//////////    //                if nums[left] + nums[right] == target {
//////////    //                    result.append([nums[left], nums[right]])
//////////    //                    /*
//////////    //                     Need to skip duplicates
//////////    //                     */
//////////    //                    while left<right && nums[left] == nums[left + 1] {
//////////    //                        left += 1
//////////    //                    }
//////////    //                    while left<right && nums[right] == nums[right - 1] {
//////////    //                        right -= 1
//////////    //                    }
//////////    //                    left += 1
//////////    //                    right -= 1
//////////    //                } else if nums[left] + nums[right] < target {
//////////    //                    left += 1
//////////    //                } else {
//////////    //                    right -= 1
//////////    //                }
//////////    //            }
//////////    //            return result
//////////    //        }
//////////    //        var i = start
//////////    //        while i < n - k + 1 {
//////////    //            if i>start {
//////////    //                if nums[i] == nums[i-1]{
//////////    //                    i += 1
//////////    //                    continue
//////////    //                }
//////////    //            }
//////////    //            let restOfArrayResult = kSum(nums, target - nums[i], i + 1, k - 1)
//////////    //            for restOfArray in restOfArrayResult {
//////////    //                result.append([nums[i]] + restOfArray)
//////////    //            }
//////////    //            i += 1
//////////    //        }
//////////    //        return result
//////////    //    }
//////////    //    func longestConsecutive(_ nums: [Int]) -> Int {
//////////    ///*
//////////    //        var numSet = Set(nums) // Using set for O(1) lookup and Set is an unordered collection
//////////    //        var longestStreak = 0
//////////    //        for num in numSet {
//////////    //            if !numSet.contains(num-1){
//////////    //                //Starting a new sequence
//////////    //                var currentNum = num
//////////    //                var currentStreak = 1
//////////    //                while numSet.contains(currentNum + 1) {
//////////    //                    currentStreak += 1
//////////    //                    currentNum += 1
//////////    //                }
//////////    //                longestStreak = max(longestStreak, currentStreak)
//////////    //            }
//////////    //        }
//////////    //        return longestStreak // This is O(n) solution, even though we are using 2 loops, the inner loop will cover the consecutive elements once and outer loop won't cover those elements again, so it is O(n) in total.
//////////    // */
//////////    //        // Using DSU to find longest consecutive sequence
//////////    //        let uniqueNums = Array(Set(nums))
//////////    //        let n = uniqueNums.count
//////////    //        var uniqueNumsToIndex: [Int: Int] = [:]
//////////    //        for (index, num) in uniqueNums.enumerated() {
//////////    //            uniqueNumsToIndex[num] = index
//////////    //        }
//////////    //
//////////    //        var dsu = DSU(n)
//////////    //
//////////    //        // Union or grouping the consecutive numbers
//////////    //        for num in uniqueNums{
//////////    //            if let nextNumIndex = uniqueNumsToIndex[num + 1] {
//////////    //                dsu.union(uniqueNumsToIndex[num]!, nextNumIndex)
//////////    //            }
//////////    //        }
//////////    //        return dsu.size.max() ?? 0 // Return the size of the largest group
//////////    //    }
//////////    /*
//////////     Longest consecutive subsequence
//////////     Kadane
//////////     K sum
//////////     */
//////////
//////////    func subarraySum(_ nums: [Int], _ k: Int) -> Int {
//////////        // needs to return the number of continuous subarrays whose sum equals to k
//////////        /*
//////////         // 1st approach is to use brute force, O(n^2) solution
//////////         var prefixSum = Array(repeating: 0, count: nums.count)
//////////         if nums.isEmpty {
//////////         return 0
//////////         }
//////////         prefixSum[0] = nums[0]
//////////         for i in 1..<nums.count {
//////////         prefixSum[i] = prefixSum[i-1] + nums[i]
//////////         }
//////////         var count = 0
//////////         for i in 0..<nums.count {
//////////         for j in i..<nums.count {
//////////         let subarray = Array(nums[i...j])  // just nums[i...j] will be a ArraySlice type and not an Array type
//////////         let sum = subarray.reduce(0, +)
//////////         if sum == k {
//////////         count += 1
//////////
//////////         }
//////////         }
//////////         }
//////////         return count
//////////         */
//////////        /*
//////////         2nd approach is to use a hashmap to store the prefix sum and its count
//////////         */
//////////        var count = 0
//////////        var currentSum = 0
//////////        var prefixSumCount: [Int: Int] = [0:1] // number of times a prefix sum has occurred
//////////        for i in 0..<nums.count {
//////////            currentSum += nums[i]
//////////            let targetSumExists = prefixSumCount[currentSum-k] ?? 0 // Check if there exists a prefix sum such that currentSum - prefixSum = k
//////////            count += targetSumExists
//////////            prefixSumCount[currentSum, default: 0] += 1 // Initialize if not present
//////////        }
//////////        return count
//////////    }
//////////
//////////    func findLongestSubarrayWithSum(_ nums: [Int], _ k: Int) -> Int {
//////////        // Returns the length of the longest subarray with sum equals to k
//////////        var count = 0
//////////        var currentSum = 0
//////////        var prefixSumIndex: [Int: Int] = [0: -1] // Maps prefix sum to its first occurrence index
//////////        for i  in 0..<nums.count {
//////////            currentSum += nums[i]
//////////            if let firstIndex = prefixSumIndex[currentSum - k] {
//////////                count = max(count, i - firstIndex) // Update count if we found a longer subarray
//////////            }
//////////            // Only add the prefix sum if it is not already present to ensure we get the longest subarray
//////////            if prefixSumIndex[currentSum] == nil {
//////////                prefixSumIndex[currentSum] = i
//////////            }
//////////        }
//////////        return count
//////////    }
//////////
//////////    func findSubarraysWithXorEqualsTok(_ nums: [Int], _ k: Int) -> Int {
//////////        // if A XOR B = k, then A = k XOR B
//////////        var count = 0
//////////        var currentXor = 0
//////////        var prefixXorCount: [Int: Int] = [0: 1] // Maps prefix XOR to its count
//////////        for i in 0..<nums.count {
//////////            currentXor ^= nums[i] // Update current XOR
//////////            let targetXor = currentXor ^ k // Find the target XOR
//////////            count += prefixXorCount[targetXor, default: 0] // Add the count of target XOR
//////////            prefixXorCount[currentXor, default: 0] += 1 // Increment the count of current XOR
//////////        }
//////////        return count
//////////    }
//////////
//////////    func lengthOfLongestSubstring(_ s: String) -> Int {
//////////        // example string: aecbcdeaf : Answer = 6 , i.e. bcdeaf
//////////        // Idea have a hashMap which tells the index , i.e last occurrence of a character and use sliding window concept
//////////        var maxLength = 0
//////////        var start = 0
//////////        var lastSeenCharacters: [Character: Int] = [:]
//////////        for (index,character) in s.enumerated() {
//////////            if let lastSeenIndex = lastSeenCharacters[character] {
//////////                start = max(start, lastSeenIndex + 1)
//////////            }
//////////            maxLength = max(maxLength, index - start + 1)
//////////            lastSeenCharacters[character] = index
//////////        }
//////////        return maxLength
//////////    }
//////////
//////////    func reverseList(_ head: ListNode?) -> ListNode? {
//////////        /* 1st method iterative solution
//////////         var prev: ListNode?
//////////         var current = head
//////////         while current != nil {
//////////         let nextNode = current?.next
//////////         current?.next = prev
//////////         prev = current
//////////         current = nextNode
//////////         }
//////////         return prev
//////////         */
//////////        //2nd method: Recursive solution
//////////        if head == nil || head?.next == nil {
//////////            return head
//////////        }
//////////        let newHead = reverseList(head?.next)
//////////        head?.next?.next = head
//////////        head?.next = nil
//////////
//////////        return newHead
//////////    }
//////////    func middleNode(_ head: ListNode?) -> ListNode? {
//////////        /* 1st method: simply finding the number of elements and traversing again to get that
//////////         var sizeOfLinkedList = 0
//////////         var currentNode: ListNode? = head
//////////         while(currentNode?.next != nil){
//////////         sizeOfLinkedList += 1
//////////         currentNode = currentNode?.next
//////////         }
//////////         var currentNodeForTraversal: ListNode? = head
//////////         for _ in 0..<(sizeOfLinkedList/2) {
//////////         currentNodeForTraversal = currentNodeForTraversal?.next
//////////         }
//////////         return currentNodeForTraversal
//////////         */
//////////        // 2nd method: 2 pointer slow/fast approach
//////////        var slow: ListNode? = head
//////////        var fast: ListNode? = head
//////////        while fast != nil && fast?.next != nil {
//////////            slow = slow?.next
//////////            fast = fast?.next?.next
//////////        }
//////////        return slow
//////////    }
//////////    /*
//////////     3rd method: Recursive solution for finding middle of Linked List
//////////     func findMiddleUtil(_ node: ListNode?, _ n: inout Int, _ mid: inout ListNode?) {
//////////     guard let node = node else {
//////////     n = (n+1) / 2   // Set n to half on reaching end
//////////     return
//////////     }
//////////     n += 1
//////////     findMiddleUtil(node.next, &n, &mid)
//////////     n -= 1
//////////     if n == 0 {
//////////     mid = node
//////////     }
//////////     }
//////////
//////////     func middleNode(_ head: ListNode?) -> ListNode? {
//////////     var n = 0
//////////     var mid: ListNode? = nil
//////////     findMiddleUtil(head, &n, &mid)
//////////     return mid
//////////     }
//////////     */
//////////    func mergeTwoLists(_ list1: ListNode?, _ list2: ListNode?) -> ListNode? {
//////////        let dummy = ListNode(0)
//////////        var tail = dummy
//////////        var node1 = list1
//////////        var node2 = list2
//////////
//////////        while let n1 = node1, let n2 = node2 {
//////////            if n1.val < n2.val {
//////////                tail.next = n1
//////////                node1 = n1.next
//////////            } else {
//////////                tail.next = n2
//////////                node2 = n2.next
//////////            }
//////////            tail = tail.next!
//////////        }
//////////
//////////        // One list may have leftovers
//////////        tail.next = node1 ?? node2
//////////        return dummy.next
//////////    }
//////////
//////////    func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
//////////        // Approach 1: Finding the length of the list and then removing the nth node from the end
//////////        // Approach 2: Using two pointers (fast and slow) to find the nth node from the end, keep the fast n steps ahead of the slow pointer, when fast reaches the end, slow will be at the nth node from the end
//////////        var dummy = ListNode(0)
//////////        dummy.next = head
//////////        var fast: ListNode? = dummy
//////////        var slow: ListNode? = dummy
//////////        for _ in 0..<n {
//////////            fast = fast?.next
//////////        }
//////////        while fast?.next != nil {
//////////            slow = slow?.next
//////////            fast = fast?.next
//////////        }
//////////        slow?.next = slow?.next?.next
//////////        return dummy.next
//////////    }
//////////
//////////    func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
//////////        /*
//////////         You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
//////////         You may assume the two numbers do not contain any leading zero, except the number 0 itself.
//////////         */
//////////        // Need to use the concept of carry here
//////////        var carry = 0
//////////        let dummyNode = ListNode(0)
//////////        var p = dummyNode
//////////        var list1 = l1, list2 = l2
//////////        while (list1 != nil || list2 != nil) {
//////////            let val1 = list1?.val ?? 0
//////////            let val2 = list2?.val ?? 0
//////////            let sum = val1 + val2 + carry
//////////            let newNodeVal = sum / 10
//////////            carry = sum % 10
//////////            p.next = ListNode(newNodeVal)
//////////            if let next = p.next {
//////////                p = next
//////////            }
//////////            list1 = list1?.next
//////////            list2 = list2?.next
//////////        }
//////////        return dummyNode.next
//////////    }
//////////
//////////    func deleteNode(_ node: ListNode?) {
//////////        var currentNode = node
//////////        var nextNode = node?.next
//////////        if currentNode != nil && nextNode != nil {
//////////            currentNode?.val = nextNode?.val ?? 0
//////////            currentNode?.next = nextNode?.next
//////////        }
//////////    }
//////////
//////////    func getIntersectionNode(_ headA: ListNode?, _ headB: ListNode?) -> ListNode? {
//////////        // find length of both the linked list
//////////        var currentA = headA, currentB = headB
//////////        var lengthA = 0, lengthB = 0
//////////        while currentA != nil {
//////////            lengthA += 1
//////////            currentA = currentA?.next
//////////        }
//////////        while currentB != nil {
//////////            lengthB += 1
//////////            currentB = currentB?.next
//////////        }
//////////        // Move the pointer of the longer list by the difference in lengths
//////////        var pointerToAdvance = lengthA > lengthB ? headA : headB
//////////        for _ in 0..<abs(lengthA - lengthB) {
//////////            pointerToAdvance = pointerToAdvance?.next
//////////        }
//////////        // Now move both pointers until they meet
//////////        var pointerA = lengthA > lengthB ? pointerToAdvance : headA
//////////        var pointerB = lengthA > lengthB ? headB : pointerToAdvance
//////////
//////////        while pointerA !== pointerB {
//////////            pointerA = pointerA?.next
//////////            pointerB = pointerB?.next
//////////        }
//////////        return pointerA // or pointerB, both will be the same at this point
//////////    }
//////////
//////////    func hasCycle(_ head: ListNode?) -> Bool {
//////////        // Method 1: Using a set to store visited nodes
//////////        /*
//////////         var visitedNodes: Set<ListNode> = []
//////////         var currentNode = head
//////////         while currentNode != nil {
//////////         if visitedNodes.contains(currentNode!) {
//////////         return true // Cycle detected
//////////         }
//////////         visitedNodes.insert(currentNode!)
//////////         currentNode = currentNode?.next
//////////         }
//////////         return false
//////////         */
//////////        // Method 2: Using Floyd's Cycle Detection Algorithm (Tortoise and Hare)
//////////        var dummyNode : ListNode? = ListNode(0)
//////////        dummyNode?.next = head
//////////        var slow = dummyNode
//////////        var fast = dummyNode
//////////        // here initially both slow and fast are same, so need to move them at least once, using do while loop
//////////        repeat {
//////////            slow = slow?.next
//////////            fast = fast?.next?.next
//////////            if fast == nil || fast?.next == nil {
//////////                return false // No cycle
//////////            }
//////////        } while slow != fast
//////////        // Meaningful cycle detected, move slow pointer to head and check if it meets fast pointer
//////////        slow = dummyNode
//////////        while slow !== fast {
//////////            slow = slow?.next
//////////            fast = fast?.next
//////////            if fast == nil || fast?.next == nil {
//////////                return false // No cycle
//////////            }
//////////        }
//////////        return true // Cycle detected : slow will point that intersection point
//////////    }
//////////
//////////    func reverseKGroup(_ head: ListNode?, _ k: Int) -> ListNode? {
//////////        // Intuition: Recursion
//////////        var currentNode: ListNode? = head
//////////
//////////        // Checking if k nodes do exist in Linked List
//////////        for _ in 0..<k {
//////////            if currentNode == nil {
//////////                return head
//////////            }
//////////            currentNode = currentNode?.next
//////////        }
//////////
//////////        // Reverse the first k nodes
//////////        var prev: ListNode? = nil
//////////        var current = head
//////////        var count = 0
//////////        for _ in 0..<k {
//////////            let nextNode = current?.next
//////////            current?.next = prev
//////////            prev = current
//////////            current = nextNode
//////////        }
//////////
//////////        // head is the new tail, previous is the new head
//////////        if let originalHead = head {
//////////            originalHead.next = reverseKGroup(current, k)
//////////        }
//////////
//////////        return prev
//////////    }
//////////
//////////    func isPalindrome(_ head: ListNode?) -> Bool {
//////////        // first find the middle of the linked list
//////////        // reverse the second half of linked list
//////////        // traverse if equal till the end : yes
//////////        // restore the order of second half to original list
//////////        if head == nil || head?.next == nil {
//////////            return true
//////////        }
//////////
//////////        var middleNode = middleNode(head)
//////////        // reverse the second half
//////////        var secondHalfHead: ListNode? = reverseList(middleNode?.next)
//////////
//////////        var firstHalfNode: ListNode? = head
//////////        var isPalindrome = true
//////////
//////////        while isPalindrome && secondHalfHead != nil {
//////////            if firstHalfNode?.val != secondHalfHead?.val {
//////////                isPalindrome = false
//////////            }
//////////            firstHalfNode = firstHalfNode?.next
//////////            secondHalfHead = secondHalfHead?.next
//////////        }
//////////
//////////        // restore the second half again
//////////        firstHalfNode?.next = reverseList(middleNode)
//////////        return isPalindrome
//////////    }
//////////
//////////    func mergeTwoSortedLinkedList(_ list1: Node?, _ list2: Node?) -> Node? {
//////////        var dummy = Node(0)
//////////        var tail: Node? = dummy
//////////        var node1 = list1
//////////        var node2 = list2
//////////        while node1 != nil && node2 != nil {
//////////            if node1!.val < node2!.val {
//////////                tail?.child = node1
//////////                node1 = node1?.child
//////////            } else {
//////////                tail?.child = node2
//////////                node2 = node2?.child
//////////            }
//////////            tail = tail?.child
//////////        }
//////////        tail?.child = node1 ?? node2
//////////        return dummy.child
//////////    }
//////////
//////////    func flatten(_ head: Node?) -> Node? {
//////////        if head == nil || head?.next == nil {
//////////                return head
//////////        }
//////////
//////////        var flattenedNext: Node? = flatten(head?.next)
//////////
//////////        head?.next = nil
//////////        let mergedLinkedListHead = mergeTwoSortedLinkedList(head, flattenedNext)
//////////
//////////        return mergedLinkedListHead
//////////    }
//////////
//////////    func lengthOfListNode(_ head: ListNode?) -> Int {
//////////        if head == nil { return 0 }
//////////        var currNode = head
//////////        var answer = 0
//////////        while currNode != nil {
//////////            answer += 1
//////////            currNode = currNode?.next
//////////        }
//////////        return answer
//////////    }
//////////
//////////    func rotateRight(_ head: ListNode?, _ k: Int) -> ListNode? {
//////////        if head == nil || head?.next == nil || k == 0 {
//////////            return head
//////////        }
//////////
//////////        let length = lengthOfListNode(head)
//////////
//////////        // Handle cases where no rotation is needed
//////////        let correctedK = k % length
//////////        if correctedK == 0 {
//////////            return head
//////////        }
//////////
//////////        // Find the old tail (last node)
//////////        var tail: ListNode? = head
//////////        while tail?.next != nil {
//////////            tail = tail?.next
//////////        }
//////////
//////////        // Find the new tail, which is at position (length - k - 1)
//////////        var newTail = head
//////////        for _ in 0..<(length - correctedK - 1) {
//////////            newTail = newTail?.next
//////////        }
//////////
//////////        // The new head is the node after the new tail
//////////        let newHead = newTail?.next
//////////
//////////        // Cut the list at the new tail
//////////        newTail?.next = nil
//////////
//////////        // Connect the old tail to the old head
//////////        tail?.next = head
//////////
//////////        return newHead
//////////    }
//////////}
//////////
//////////extension ListNode: Hashable {
//////////    // Hashable conformance
//////////    public func hash(into hasher: inout Hasher) {
//////////        hasher.combine(ObjectIdentifier(self))  // Hash based on memory address
//////////    }
//////////
//////////    public static func == (lhs: ListNode, rhs: ListNode) -> Bool {
//////////        return lhs === rhs  // Check if same object (reference equality)
//////////    }
//////////}
//////////
//////////public class Node {
//////////    public var val: Int
//////////    public var next: Node?
//////////    public var child: Node?
//////////    public init(_ val: Int) { self.val = val; self.next = nil; self.child = nil; }
//////////}
//////////
//////////public class ListNode {
//////////    public var val: Int
//////////    public var next: ListNode?
//////////    public init() { self.val = 0; self.next = nil; }
//////////    public init(_ val: Int) { self.val = val; self.next = nil; }
//////////    public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
//////////}
//////////
//////////class DSU {
//////////    var parent: [Int] // Parent of each node
//////////    var size: [Int] // Size of each group
//////////    init(_ parent: [Int], _ size: [Int]) {
//////////        self.parent = parent
//////////        self.size = size
//////////    }
//////////
//////////    init(_ n: Int) {
//////////        self.parent = Array(0..<n) // Each node is its own parent initially
//////////        self.size = Array(repeating: 1, count: n) // Each node is its own parent and size is 1 initially
//////////    }
//////////
//////////    func find(_ x: Int) -> Int {
//////////        if parent[x] != x {
//////////            parent[x] = find(parent[x])
//////////        }
//////////        return parent[x]
//////////    }
//////////    func union(_ x: Int, _ y: Int) {
//////////        let parentOfX = find(x)
//////////        let parentOfY = find(y)
//////////        if parentOfX == parentOfY {
//////////            return
//////////        }
//////////        if size[parentOfX] < size[parentOfY] {
//////////            parent[parentOfX] = parentOfY
//////////            size[parentOfY] += size[parentOfX]
//////////        } else {
//////////            parent[parentOfY] = parentOfX
//////////            size[parentOfX] += size[parentOfY]
//////////        }
//////////    }
//////////}
//////////
//////////let solution = Solution()
////////////let nums = [1,0,-1,0,-2,2]
////////////let target = 0
////////////solution.fourSum(nums, target)
//////////
//////////
////////
//////////public class Node {
//////////  public var val: Int
//////////  public var next: Node?
//////////  public var random: Node?
//////////  public init(_ val: Int) {
//////////      self.val = val
//////////      self.next = nil
//////////      self.random = nil
//////////  }
//////////}
//////////
//////////extension Node: Hashable {
//////////    public func hash(into hasher: inout Hasher) {
//////////        hasher.combine(ObjectIdentifier(self))
//////////    }
//////////
//////////    public static func == (lhs: Node, rhs: Node) -> Bool {
//////////        return lhs === rhs // reference equality
//////////    }
//////////}
//////////
//////////class Solution {
//////////    func copyRandomList(_ head: Node?) -> Node? {
//////////        guard let head = head else { return nil }
//////////
//////////        /* 1. O(n^2): Approach
//////////         var originalNodes: [Node] = []
//////////         var copyNodes: [Node] = []
//////////         var node : Node? = head
//////////         while let currNode = node {
//////////         originalNodes.append(currNode)
//////////         copyNodes.append(Node(currNode.val))
//////////         node = currNode.next
//////////         }
//////////
//////////         let n = originalNodes.count
//////////         for i in 0..<n {
//////////         if i < n-1 {
//////////         copyNodes[i].next = copyNodes[i+1]
//////////         }
//////////         if let randomOriginalNode = originalNodes[i].random {
//////////         // finding this randomOriginalNode in originalNode array
//////////         if let idx = originalNodes.firstIndex(where: { $0 === randomOriginalNode }){
//////////         copyNodes[i].random = copyNodes[idx]
//////////         }
//////////         }
//////////         }
//////////         return copyNodes.first
//////////         }
//////////         */
//////////        // Using Hashmap
//////////        var oldNodeToNewNodeMap: [Node: Node] = [:]
//////////        var currentNode: Node? = head
//////////        while let node = currentNode {
//////////            let newNode = Node(node.val)
//////////            oldNodeToNewNodeMap[node] = newNode
//////////            currentNode = node.next
//////////        }
//////////        // Set the next and random pointer now
//////////        currentNode = head
//////////        while let node = currentNode {
//////////            if let newNode = oldNodeToNewNodeMap[node] {
//////////                newNode.next = node.next == nil ? nil : oldNodeToNewNodeMap[node.next!]
//////////                newNode.random = node.random == nil ? nil : oldNodeToNewNodeMap[node.random!]
//////////            }
//////////            currentNode = node.next
//////////        }
//////////        return oldNodeToNewNodeMap[head]
//////////    }
//////////}
////////
////////class Solution {
////////    //    func trap(_ height : [Int]) -> Int {
////////    //        let n = height.count
////////    //        //1. Recursive approach
////////    //        //2. Add memoization
////////    //       var leftMax = [Int:Int]()
////////    //       var rightMax = [Int:Int]()
////////    //        func maxLeft(_ index : Int) -> Int {
////////    //            if index == 0 {
////////    //                return 0
////////    //            }
////////    //    if let val = leftMax[index] { return val }
////////    //            let val = max(height[index-1], maxLeft(index-1))
////////    //            leftMax[index] = val
////////    //            return val
////////    //        }
////////    //        func maxRight(_ index: Int)-> Int {
////////    //            if index == n-1 {
////////    //                return 0
////////    //            }
////////    //    if let val = rightMax[index] {return val}
////////    //            let val = max(height[index+1], maxRight(index+1))
////////    //            rightMax[index] = val
////////    //            return val
////////    //        }
////////    //        var total = 0
////////    //        for i in 0..<n {
////////    //            let val = (min(maxLeft(i),maxRight(i))) - height[i]
////////    //            if val > 0 {
////////    //                total = total + val
////////    //            }
////////    //        }
////////    //        return total
////////    //    }
////////    //    func trap(_ height: [Int]) -> Int {
////////    //        //3. Using Dynamic Programming
////////    //        let n = height.count
////////    //        var leftMax: [Int] = Array(repeating: 0, count: n)
////////    //        var rightMax: [Int] = Array(repeating: 0, count: n)
////////    //        leftMax[0] = height[0]
////////    //        for i in 1..<n {
////////    //            leftMax[i] = max( leftMax[i-1], height[i-1] )
////////    //        }
////////    //
////////    //
////////    //        //through will go till 0
////////    //        for i in stride(from:n-2, through:0, by: -1){
////////    //            rightMax[i] = max(rightMax[i+1], height[i+1])
////////    //        }
////////    //
////////    //
////////    //        var total = 0
////////    //        for i in 0..<n{
////////    //            let val = min(leftMax[i], rightMax[i])-height[i]
////////    //            if val > 0 {
////////    //                total = total + val
////////    //            }
////////    //        }
////////    //        return total
////////    //
////////    //    }
////////    //    func trap(_ height: [Int]) -> Int {
////////    //        //4. Using two pointer approach
////////    //        let n = height.count
////////    //        var left = 0, right = n-1, maxLeft = 0, maxRight = 0, total = 0
////////    //        while left < right{
////////    //            if height[left]<height[right] {
////////    //                // left index storage
////////    //                if height[left] >= maxLeft {
////////    //                    maxLeft = height[left]
////////    //                } else {
////////    //                    total = total + (maxLeft - height[left])
////////    //                }
////////    //                left += 1
////////    //            } else {
////////    //                // right index storage
////////    //                if height[right] >= maxRight {
////////    //                    maxRight = height[right]
////////    //                } else {
////////    //                    total = total + (maxRight - height[right])
////////    //                }
////////    //                right -= 1
////////    //            }
////////    //        }
////////    //        return total
////////    //    }
////////    /*
////////     The Intuition Behind the Stack
////////     Think of the stack as a way to keep track of potential left walls of a container that could hold water.
////////
////////     What does the stack hold? The stack stores the indices of the bars. We always maintain the property that the heights of the bars corresponding to the indices in the stack are in decreasing or equal order from bottom to top. For example, stack = [index_of_height_5, index_of_height_3, index_of_height_2].
////////
////////     When do we find water? We find trapped water when we encounter a bar that is taller than the bar at the top of the stack. This new, taller bar acts as a right wall.
////////
////////     The bar we just popped from the stack (top) becomes the bottom of the container.
////////
////////     The new top of the stack (after popping) becomes the left wall.
////////
////////     Why calculate h * w? Once we have a left wall, a bottom, and a right wall, we've defined a "valley" or a "V" shape. The water trapped in this valley fills up like a rectangle.
////////
////////     h (Height of water): The water can only fill up to the height of the shorter of the two walls (left and right). So, h = min(height[left_wall], height[right_wall]) - height[bottom]. We subtract the bottom's height because that space is already occupied by the bar itself.
////////
////////     w (Width of water): This is simply the distance between the left and right walls. w = index_of_right_wall - index_of_left_wall - 1.
////////     */
////////    //    func trap(_ height: [Int]) -> Int {
////////    //        var stack = [Int]()
////////    //        var total = 0
////////    //        for i in 0..<height.count {
////////    //            //  i is the right wall
////////    //            while !stack.isEmpty && height[i]>height[stack.last!] {
////////    //                let top = stack.removeLast() // This is dip part of the valley
////////    //                if stack.isEmpty { break }
////////    //                let left = stack.last!
////////    //                let height = min(height[i], height[left])-height[top]
////////    //                let width = i - left - 1
////////    //                total += height * width
////////    //            }
////////    //            stack.append(i)
////////    //        }
////////    //        return total
////////    //    }
////////
////////    func removeDuplicates(_ nums: inout [Int]) -> Int {
////////        var start = 0, current = 0
////////        let n = nums.count
////////        while current < n {
////////            if nums[start] != nums[current] && start < nums.count - 2 {
////////                nums[start+1] =  nums[current]
////////                start += 1
////////            }
////////            current += 1
////////        }
////////        print(nums)
////////        return start+1
////////    }
////////
////////    func jobScheduling(_ jobs: [[Int]]) -> (jobCount: Int, maxProfit: Int) {
////////        var jobs = jobs.sorted{ $0[2] > $1[2]}
////////        let maxDeadline = jobs.map{$0[1]}.max() ?? 0
////////        var time = Array(repeating: false, count: maxDeadline+1)
////////        var profit = 0
////////        var jobCount = 0
////////        for job in jobs {
////////            let deadline = job[1]
////////            let profitForJob = job[2]
////////
////////            for timeSlot in stride(from: deadline, to: 0, by: -1){
////////                if !time[timeSlot] {
////////                    jobCount += 1
////////                    profit += profitForJob
////////                    time[timeSlot] = true
////////                    break
////////                }
////////            }
////////        }
////////        return (jobCount, profit)
////////    }
////////
////////    func fractionalKnapSack(_ value:[Int], _ weight:[Int], _ capacity:Int) -> Double {
////////        // Greed approach: To maximize the value/weight ratio, value bigger and weight smaller
////////        var items = [(ratio: Double, value: Int, weight: Int)]()
////////        for i in 0..<value.count {
////////            let ratio = Double(value[i])/Double(weight[i])
////////            items.append((ratio, value[i], weight[i]))
////////        }
////////        items.sort{$0.ratio>$1.ratio}
////////        var profit: Double = 0.0
////////        var remainingCapacity = capacity
////////        for item in items {
////////            if remainingCapacity == 0 { break }
////////            if item.weight <= remainingCapacity {
////////                // take full
////////                profit += Double(item.value)
////////                remainingCapacity -= item.weight
////////            } else {
////////                let holdingCapacity = Double(item.weight)/Double(remainingCapacity)
////////                profit += Double(item.value) * holdingCapacity
////////                remainingCapacity = 0
////////            }
////////        }
////////        return profit
////////    }
////////
////////    func coinChangeRecursive(_ coins: [Int], _ amount: Int) -> Int {
////////        if amount == 0 {return 0}
////////        if amount < 0 {return -1}
////////
////////        var minCoins = Int.max
////////        for coin in coins {
////////            let minimumCoinsRequiredIfConsideringCoin = coinChangeRecursive(coins, amount - coin)
////////            if minimumCoinsRequiredIfConsideringCoin >= 0 && minCoins>minimumCoinsRequiredIfConsideringCoin  {
////////                minCoins = 1+minimumCoinsRequiredIfConsideringCoin
////////            }
////////        }
////////        return (minCoins == Int.max) ? -1: minCoins
////////    }
////////
////////    func coinChangeRecursiveWithMemoization(_ coins: [Int], _ amount: Int) -> Int {
////////        var memo = [Int:Int]() //minimum number of coins required(value) of this amount (key)
////////        func dp(_ rem: Int) -> Int {
////////            if rem == 0 {return 0}
////////            if rem < 0 {return -1}
////////            if let cached = memo[rem] { return cached }
////////            var minCoins = Int.max
////////            for coin in coins {
////////                let minimumCoinsRequiredIfConsideringCoin = dp(rem - coin)
////////                if minimumCoinsRequiredIfConsideringCoin >= 0 && minCoins>minimumCoinsRequiredIfConsideringCoin  {
////////                    minCoins = 1+minimumCoinsRequiredIfConsideringCoin
////////                }
////////            }
////////            memo[rem] = (minCoins == Int.max) ? -1: minCoins
////////            return memo[rem]!
////////        }
////////        return dp(amount)
////////    }
////////
////////    func coinChangeDP(_ coins: [Int], _ amount: Int) -> Int {
////////        if amount == 0 { return 0 }
////////        var dp = Array(repeating: Int.max, count: amount + 1)
////////        dp[0] = 0
////////        for total in 1...amount {
////////            for coin in coins {
////////                if total >= coin && dp[total - coin] != Int.max {
////////                    dp[total] = min(dp[total], 1+dp[total-coin])
////////                }
////////            }
////////        }
////////        return dp[amount] == Int.max ? -1 : dp[amount]
////////    }
////////
////////    // assuming canonical denomination
////////    func coingChangeGreedy( _ coins: [Int], _ amount: Int) -> Int{
////////        let sortedCoins = coins.sorted(by: >)
////////        var count  = 0
////////        var amountLeft = amount
////////        for coin in sortedCoins {
////////            if amountLeft == 0 {break}
////////            let use = amountLeft/coin
////////            count += use
////////            amountLeft -= use*coin
////////        }
////////        return amountLeft == 0 ? count : -1
////////    }
////////
////////    func findContentChildren(_ g: [Int], _ s: [Int]) -> Int {
////////        var sortedGreed = g.sorted{$0<$1}
////////        var sortedSize = s.sorted{$0<$1}
////////        var gSize = g.count, sSize = s.count
////////        var i=0, j=0
////////        var answer = 0
////////        while i<gSize && j<sSize {
////////            if sortedGreed[i] <= sortedSize[j]{
////////                answer += 1
////////                //child is satisfied
////////                i += 1
////////            }
////////            j += 1
////////        }
////////        return answer
////////    }
////////
////////    func subsetsWithDup(_ nums: [Int]) -> [[Int]] {
////////       // Pure recursive  solution vibe, either consider the element or not
////////        var result = [[Int]]()
////////        var current = [Int]()
////////        let nums = nums.sorted()
////////        let n = nums.count
////////        func dfs(_ index: Int) {
////////            result.append(current)
////////            for i in index..<n {
////////                if i != index && nums[i] == nums[i-1]{ continue } // i!=index makes sure that it's not the first time we are encountering this element to make the array of this size
////////                current.append(nums[i])
////////                dfs(i+1)
////////                current.removeLast()
////////            }
////////        }
////////        dfs(0)
////////        return Array(result)
////////    }
////////
////////}
////////
////////let obj = Solution()
////////var array = [0,0,1,1,1,2,2,3,3,4]
////////let k = obj.removeDuplicates(&array)
////////print(k)
////////
//////
//////class Solution {
//////    // This prints all the subsequences : 2^n
//////    // func subsetsWithDup(_ nums: [Int]) -> [[Int]] {
//////    //    // Pure recursive  solution vibe, either consider the element or not
//////    //     var result = [[Int]]()
//////    //     var current = [Int]()
//////    //     let n = nums.count
//////    //     func dfs(_ index: Int){
//////    //         if index == n {
//////    //             result.append(current)
//////    //             return
//////    //         }
//////    //         // take the element
//////    //         current.append(nums[index])
//////    //         dfs(index+1)
//////    //         //skip the last element
//////    //         current.removeLast()
//////    //         dfs(index+1)
//////    //     }
//////    //     dfs(0)
//////    //     return result
//////    // }
//////    /*
//////    Below function will print the unique subsequences but here for example
//////    [4,4,4,1,4]
//////    it gives output:
//////    [[4],[4,4,4],[4,4,4,4],[1,4],[],[4,1,4],[4,1],[4,4,1,4],[4,4],[4,4,1],[1],[4,4,4,1],  [4,4,4,1,4]]
//////    But as you can see though if you want to preseve the order then yes [4,1,4] and [4,4,1] are different subsequences
//////    But if you don't want to presever the order then they are both same, thus we want to sort the array first so that all similar elements gets grouped together
//////    */
//////    // func subsetsWithDup(_ nums: [Int]) -> [[Int]] {
//////    //    // Pure recursive  solution vibe, either consider the element or not
//////    //     var result = Set<[Int]>()
//////    //     var current = [Int]()
//////    //     // let nums = nums.sorted()
//////    //     let n = nums.count
//////    //     func dfs(_ index: Int){
//////    //         if index == n {
//////    //             result.insert(current)
//////    //             return
//////    //         }
//////    //         // take the element
//////    //         current.append(nums[index])
//////    //         dfs(index+1)
//////    //         //skip the last element
//////    //         current.removeLast()
//////    //         dfs(index+1)
//////    //     }
//////    //     dfs(0)
//////    //     return Array(result)
//////    // }
//////    // func subsetsWithDup(_ nums: [Int]) -> [[Int]] {
//////    //    // Pure recursive  solution vibe, either consider the element or not
//////    //     var result = Set<[Int]>()
//////    //     var current = [Int]()
//////    //     let nums = nums.sorted()
//////    //     let n = nums.count
//////    //     func dfs(_ index: Int){
//////    //         if index == n {
//////    //             result.insert(current)
//////    //             return
//////    //         }
//////    //         // take the element
//////    //         current.append(nums[index])
//////    //         dfs(index+1)
//////    //         //skip the last element
//////    //         current.removeLast()
//////    //         dfs(index+1)
//////    //     }
//////    //     dfs(0)
//////    //     return Array(result)
//////    // }
//////
//////    // Can also solve without the Set<Int>(), solving within the recursion
//////    // func subsetsWithDup(_ nums: [Int]) -> [[Int]] {
//////    //    // Pure recursive  solution vibe, either consider the element or not
//////    //     var result = [[Int]]()
//////    //     var current = [Int]()
//////    //     let nums = nums.sorted()
//////    //     let n = nums.count
//////    //     func dfs(_ index: Int) {
//////    //         result.append(current)
//////    //         for i in index..<n {
//////    //             if i != index && nums[i] == nums[i-1]{ continue } // i!=index makes sure that it's not the first time we are encountering this element to make the array of this size
//////    //             current.append(nums[i])
//////    //             dfs(i+1)
//////    //             current.removeLast()
//////    //         }
//////    //     }
//////    //     dfs(0)
//////    //     return Array(result)
//////    // }
//////
//////    //Can also solve using bfs i.e. creating subsets of length 1 first, then 2 ...
//////
//////    func subsetsWithDup(_ nums: [Int]) -> [[Int]] {
//////        var queue : [([Int],Int)] = [([],0)]
//////        let n = nums.count
//////        let nums = nums.sorted(by: <)
//////        var result = [[Int]]()
//////        while !queue.isEmpty {
//////            let size = queue.count
//////            var (current, nextIndex) = queue.removeFirst()
//////            print((current,nextIndex))
//////            result.append(current)
//////            for i in nextIndex..<n {
//////                if i != nextIndex && nums[i] == nums[i-1] {continue}
//////                var next = current
//////                next.append(nums[i])
//////                queue.append((next,i+1))
//////            }
//////        }
//////        return result
//////    }
//////
//////    //Can also solve this question using bit manipulation
//////    // func subsetsWithDup(_ nums: [Int]) -> [[Int]] {
//////    //     var result = Set<[Int]>()
//////    //     let n = nums.count
//////    //     let nums = nums.sorted()
//////    //     var maxPossibleCombination = 1<<n
//////    //     for possibility in 0..<maxPossibleCombination {
//////    //         var subset = [Int]()
//////    //         for j in 0..<n {
//////    //             if (possibility & (1<<j)) != 0 {
//////    //                 subset.append(nums[j])
//////    //             }
//////    //         }
//////    //         result.insert(subset)
//////    //     }
//////    //     return Array(result)
//////    // }
//////
//////}
////
////class Solution {
//////    func combinationSum(_ candidates: [Int], _ target: Int) -> [[Int]] {
//////        var result = [[Int]]()
//////        var current = [Int]()
//////        let n = candidates.count
//////        func helper(_ index: Int, _ currentSum: Int){
//////            if currentSum == target {
//////                result.append(current)
//////                return
//////            }
//////            if currentSum > target || index >= n {
//////                return
//////            }
//////            // Case 1: Considering taking candidates[index] in the answer and finding both the branches
//////            current.append(candidates[index])
//////            helper(index, currentSum + candidates[index])
//////            // let answerWhenConsideringCurrElementAndDifferentBranch = helper(index+1, currentSum + target) -> This case will be covered within the next case only
//////            current.removeLast()
//////            // Case 2: Not considering current element
//////            helper(index+1, currentSum)
//////        }
//////        helper(0,0)
//////        return result
//////    }
//////
////    // Implementing using memoization
////    func combinationSum(_ candidates: [Int], _ target: Int) -> [[Int]] {
////        var current = [Int]()
////        let n = candidates.count
////        var memo = [String: [[Int]]]()
////        func helper(_ index: Int, _ currentSum: Int) -> [[Int]] {
////            let key = "\(index),\(currentSum)"
////            var result = [[Int]]()
////            if let cachedResult = memo[key] {
////                return cachedResult
////            }
////            if currentSum == target { return [[]] }
////            if currentSum > target || index >= n { return [] }
////
////            for path in helper(index, currentSum+candidates[index]){
////                result.append(path + [candidates[index]])
////            }
////
////            for path in helper(index+1, currentSum){
////                result.append(path)
////            }
////            memo[key] = result
////            return result
////        }
////        return helper(0,0)
////    }
////
////    func combinationSumDP(_ candidates: [Int], _ target: Int) -> [[Int]] {
////        var dp = Array(repeating: [[Int]](), count: target + 1)
////        dp[0] = [[]]
////        for cand in candidates {
////            if cand <= target { // <--- fixes the runtime error
////                for t in cand...target {
////                    for comb in dp[t - cand] {
////                        dp[t].append(comb + [cand])
////                    }
////                }
////            }
////        }
////        return dp[target]
////    }
////}
//
//class Solution {
//    //    func partition(_ s: String) -> [[String]] {
//    //        var result = [[String]]()
//    //        var current = [String]()
//    //        let n = s.count
//    //        func helper(_ index: Int) {
//    //            if index == n {
//    //                result.append(current)
//    //                return
//    //            }
//    //            for i in index..<n {
//    //                // Need to make sure substring start...i is a Palindrome
//    //                let substring = String(s[s.index(s.startIndex, offsetBy: index)..<s.index(s.startIndex, offsetBy: i+1)])
//    //                if isPalindrome(substring) {
//    //                    current.append(substring)
//    //                    helper(i+1)
//    //                    current.removeLast()
//    //                }
//    //            }
//    //        }
//    //        helper(0)
//    //        return result
//    //    }
//    // Can apply memoization to above code
//    //    func partition(_ s: String) -> [[String]] {
//    //        var memo = [String: [[String]]]()
//    //        let n = s.count
//    //        func helper(_ index: Int) -> [[String]] {
//    //            let key = "\(index)"
//    //            if let cachedResult = memo[key] {
//    //                return cachedResult
//    //            }
//    //            var result = [[String]]()
//    //            if index == n {
//    //                return [[]]
//    //            }
//    //            for i in index..<n {
//    //                let substring = String(s[s.index(s.startIndex, offsetBy: index)..<s.index(s.startIndex, offsetBy: i+1)])
//    //                if isPalindrome(substring) {
//    //                    var tempResult = helper(i+1)
//    //                    for path in tempResult {
//    //                        var newPath = path
//    //                        newPath.insert(substring, at: 0)
//    //                        result.append(newPath)
//    //                    }
//    //                }
//    //            }
//    //            memo[key] = result
//    //            return result
//    //        }
//    //        return helper(0)
//    //    }
//    // can apply iterative Dynamic Programming to above code
//    func partition(_ s: String) -> [[String]] {
//        let n = s.count
//        let arr = Array(s)
//        // dp[i]: all palindromic partitions of s[0..<i]
//        var dp = Array(repeating: [[String]](), count: n + 1)
//        dp[0] = [[]] // one way to partition the empty string
//        
//        // Precompute palindrome substrings, isPal[i][j] : if substring from i...j is a palindrome or not
//        var isPal = Array(repeating: Array(repeating: false, count: n), count: n)
//        for length in 1...n {
//            for start in 0...(n - length) {
//                let end = start + length - 1
//                if arr[start] == arr[end] && (length <= 2 || isPal[start + 1][end - 1]) {
//                    isPal[start][end] = true
//                }
//            }
//        }
//        
//        for i in 1...n { // for each possible end index (exclusive)
//            for j in 0..<i { // for each possible start index
//                if isPal[j][i - 1] {
//                    let word = String(arr[j..<i])
//                    for partition in dp[j] {
//                        dp[i].append(partition + [word])
//                    }
//                }
//            }
//        }
//        return dp[n]
//    }
//    
//    
//    func isPalindrome(_ s: String) -> Bool {
//        var left = 0
//        var right = s.count - 1
//        while left < right {
//            if s[s.index(s.startIndex, offsetBy: left)] != s[s.index(s.startIndex, offsetBy: right)] {
//                // for s[left] != s[right] -> 'subscript(_:)' is unavailable: cannot subscript String with an Int, use a String.Index instead.
//                return false
//            }
//            left += 1
//            right -= 1
//        }
//        return true
//    }
//    
//    func getPermutation(_ n: Int, _ k: Int) -> String {
//        var fact = 1
//        var nums = (1...n).map{String($0)}
//        var answer = ""
//        for i in 1..<n{
//            fact = fact * i
//        }
//        var k = k-1
//        while(true){
//            answer = answer + String(nums[k/fact])
//            nums.remove(at: k/fact)
//            if k == 0 { break }
//            k = k%fact
//            fact = fact/nums.count
//        }
//        return answer
//    }
//    
//    //    func solveSudoku(_ board: inout [[Character]]) {
//    //        let n = board.count
//    //
//    //        func isSafe(_ row: Int, _ col: Int, _ num: Character) -> Bool {
//    //            for k in 0..<n {
//    //                if board[row][k] == num {
//    //                    return false
//    //                }
//    //                if board[k][col] == num {
//    //                    return false
//    //                }
//    //            } // need to check till n because there's a possibility of having a character present on an index we have yet not explored yet
//    //
//    //            // need to check which 3 x 3 block this row, col exists in
//    //            let boxRowStart = (row/3) * 3 // example : row = 7, col = 6 (7/3 will give 2 and 2*3 will give 6 i.e. last box slot)
//    //            let boxColStart = (col/3) * 3
//    //            for i in boxRowStart..<boxRowStart+3 {
//    //                for j in boxColStart..<boxColStart+3 {
//    //                    if board[i][j] == num {
//    //                        return false
//    //                    }
//    //                }
//    //            }
//    //            return true
//    //        }
//    //
//    //        // we will solve this cell by cell from top-left to right-bottom (row major)
//    //        func backtrack(_ row: Int,_ col: Int) -> Bool {
//    //            if row == n {
//    //                return true // solved
//    //            }
//    //
//    //            let nextRow = col == n-1 ? row+1 : row //If col is n-1 i.e. last col, switch to new row
//    //            let nextCol = col == n-1 ? 0 : col+1
//    //
//    //            if board[row][col] != "." {
//    //                return backtrack(nextRow, nextCol) //already filled
//    //            }
//    //
//    //            for number in 1...9 {
//    //                let characterNumber = Character(String(number))
//    //                if isSafe(row, col, characterNumber){
//    //                    board[row][col] = characterNumber
//    //                    if backtrack(nextRow, nextCol){
//    //                        return true // we don't need to find and explore further solutions since we have gotten our desired result
//    //                    }
//    //                    board[row][col] = "." //backtrack
//    //                }
//    //            }
//    //            return false
//    //        }
//    //
//    //        _ = backtrack(0, 0)
//    //    }
//    
//    // can use hashing to find isSafe
//    func solveSudoku(_ board: inout [[Character]]) {
//        let n = board.count
//        
//        // var rowUsed[i][d] will have a boolean value i.e if this digit d exist in this ith row or not
//        var rowUsed : [[Bool]] = Array(repeating: Array(repeating: false, count: 10), count: n)
//        var colUsed : [[Bool]] = Array(repeating: Array(repeating: false, count: 10), count: n)
//        var boxUsed : [[Bool]] = Array(repeating: Array(repeating: false, count: 9), count: n) // var boxUsed[b][d] will have boolean value i.e. if digit d exist in this specific box or not
//        
//        for i in 0..<n {
//            for j in 0..<n {
//                if board[i][j] != "." {
//                    let num = Int(String(board[i][j]))!
//                    rowUsed[i][num] = true
//                    colUsed[j][num] = true
//                    let b = (i/3) * 3 + (j/3)
//                    boxUsed[b][num] = true
//                }
//            }
//        }
//        
//        
//        // we will solve this cell by cell from top-left to right-bottom (row major)
//        func backtrack(_ row: Int,_ col: Int) -> Bool {
//            if row == n {
//                return true // solved
//            }
//            
//            let nextRow = col == n-1 ? row+1 : row //If col is n-1 i.e. last col, switch to new row
//            let nextCol = col == n-1 ? 0 : col+1
//            
//            if board[row][col] != "." {
//                return backtrack(nextRow, nextCol) //already filled
//            }
//            
//            for number in 1...9 {
//                let characterNumber = Character(String(number))
//                if rowUsed[row][number] == false && colUsed[col][number] == false && boxUsed[(row/3) * 3 + (col/3)][number] == false{
//                    board[row][col] = characterNumber
//                    
//                    let b = (row/3) * 3 + (col/3)
//                    
//                    rowUsed[row][number] = true
//                    colUsed[col][number] = true
//                    boxUsed[b][number] = true
//                    
//                    if backtrack(nextRow, nextCol){
//                        return true // we don't need to find and explore further solutions since we have gotten our desired result
//                    }
//                    
//                    rowUsed[row][number] = false
//                    colUsed[col][number] = false
//                    boxUsed[b][number] = false
//                    
//                    board[row][col] = "." //backtrack
//                }
//            }
//            return false
//        }
//        
//        _ = backtrack(0, 0)
//    }
//    
//    
//    // edges: list of undirected edges [u, v]
//    // m: maximum number of colors allowed
//    // n: number of vertices (0..n-1)
//    
//    func graphColoringWithAtmostMColors(_ edges: [[Int]], _ m: Int, _ n: Int ) -> Bool {
//        // reference : https://www.youtube.com/watch?v=wuVwUK25Rfc&list=PLgUwDviBIf0p4ozDR_kJJkONnb1wdx2Ma&index=60
//        // building adjacency list first
//        var adj = Array(repeating: [Int](), count: n)
//        for edge in edges {
//            let u = edge[0]
//            let v = edge[1]
//            adj[u].append(v)
//            adj[v].append(u)
//        }
//        
//        var color = Array(repeating: 0, count: n)
//        //color[u] will represent which color is assigned to uth vertex as of now
//        // color[u] == 0, means uncolored and 1...m are the assigned color
//        
//        func isSafe(_ u: Int, _ c: Int) -> Bool {
//            for v in adj[u] {
//                if c == color[v] { return false }
//            }
//            return true
//        }
//        
//        func backTracking(_ u: Int) -> Bool {
//            if u == n {
//                return true // have successfully assigned the colors to all the nodes
//            }
//            // lets check for each color and rather than traversing the graph let's travel to each of the vertices and check it's adjacent
//            
//            for c in 1...m {
//                if color[u] == 0 && isSafe(u, c) {
//                    color[u] = c
//                    if backTracking(u+1) {
//                        return true
//                    }
//                    color[u] = 0
//                }
//            }
//            return false
//        }
//        return backTracking(0)
//    }
//    
//    func findPath(_ grid: [[Int]]) -> [String] {
//        // your code goes here
//        /*
//         Problem Statement: Given a grid of dimensions n x n. A rat is placed at coordinates (0, 0) and wants to reach at coordinates (n-1, n-1).
//         Find all possible paths that rat can take to travel from (0, 0) to (n-1, n-1). The directions in which rat can move are 'U' (up) , 'D' (down) , 'L' (left) , 'R' (right).
//         The value 0 in grid denotes that the cell is blocked and rat cannot use that cell for travelling, whereas value 1 represents that rat can travel through the cell. If the cell (0, 0) has 0 value, then mouse cannot move to any other cell.
//         Note :
//         In a path no cell can be visited more than once.
//         If there is no possible path then return empty vector.
//         */
//        let n = grid.count
//        
//        var answer : [String] = []
//        var currentPath : String = ""
//        
//        var visited = Array(repeating: Array(repeating: false, count: n), count: n)
//        
//        func isSafe(_ row: Int, _ col: Int) -> Bool {
//            return row >= 0 && row < n && col >= 0 && col < n && grid[row][col] == 1 && !visited[row][col]
//        }
//        
//        func backTrack(_ row: Int,_ col: Int,_ ch: String) {
//            if !isSafe(row, col) {
//                return
//            }
//            
//            currentPath.append(ch)
//            visited[row][col] = true
//            
//            if row == n-1 && col == n-1 {
//                answer.append(currentPath)
//            } else {
//                
//                // will explore all four directions, for example if we go down i.e. (row+1,c) and there exists the paths to reach (n-1,n-1) then for all the paths add the "D" at first of that path
//                backTrack(row+1, col, "D")
//                backTrack(row, col+1, "R")
//                backTrack(row-1, col, "U")
//                backTrack(row, col-1, "L")
//            }
//            
//            if !currentPath.isEmpty {
//                currentPath.removeLast()
//            }
//            visited[row][col] = false
//        }
//        if grid[0][0] == 1 {
//            backTrack(0,0,"")
//        }
//        return answer
//    }
//    
//    // 1st approach
//    //    func canSegment(_ s: String, _ wordDict: [String], _ idx: Int) -> Bool {
//    //        if s.isEmpty { return true }
//    //        if idx == wordDict.count { return false }
//    //
//    //        let word = wordDict[idx]
//    //        var range = s.range(of: word)
//    //        while let r = range {
//    //            // Remove this occurrence of word in s
//    //            var newStr = s
//    //            newStr.removeSubrange(r)
//    //            // Recur: Try to segment what's left, using the next word
//    //            if canSegment(newStr, wordDict, idx + 1) { return true }
//    //            // Find another occurrence of this word further along
//    //            range = s.range(of: word, range: r.upperBound..<s.endIndex)
//    //        }
//    //        // Try skipping this word entirely
//    //        return canSegment(s, wordDict, idx + 1)
//    //    }
//    
//    //2nd approach: Iterating through s and checking for prefix it exists in wordDict
//    //    func canSegment(_ s: String, _ wordDict: [String]) -> Bool {
//    //        let wordSet = Set(wordDict)
//    //        let sArr = Array(s)
//    //        let n = sArr.count
//    //        var memo = [Int: Bool]()
//    //        func dfs(_ start: Int) -> Bool {
//    //            if start == n {
//    //                return true
//    //            }
//    //            if let cached = memo[start] {
//    //                return cached
//    //            }
//    //            for end in start+1..<n {
//    //                let word = String(sArr[start...end])
//    //                if wordSet.contains(word) && dfs(end+1) {
//    //                    memo[start] = true
//    //                    return true
//    //                }
//    //            }
//    //            memo[start] = false
//    //            return false
//    //        }
//    //       return dfs(0)
//    //    }
//    
//    //3rd approach: Iterative DP
//    func canSegment(_ s: String, _ wordDict: [String]) -> Bool {
//        let wordSet = Set(wordDict)
//        let sArr = Array(s)
//        let n = sArr.count
//        var dp = Array(repeating: false, count: n+1)
//        //dp[i] means if the prefix s[0..<i] can be segmented or not
//        dp[0] = true // comparing to 2nd approach i means start here
//        for i in 1...n {
//            for j in 0..<i {
//                let word = String(sArr[j..<i])
//                if dp[j] && wordSet.contains(word) {
//                    dp[i] = true
//                    break
//                }
//            }
//        }
//        return dp[n]
//    }
//    
//    //    func nthRoot(_ m: Int, _ n:Int) -> Int {
//    //        var result = 1
//    //        var factor = 2
//    //        var m = m
//    //
//    //        while factor * factor <= m {
//    //            var power = 0
//    //            while m % factor == 0 {
//    //                m /= factor
//    //                power += 1
//    //            }
//    //            if power > 0 {
//    //                if (power % n) != 0 {
//    //                    return -1
//    //                }
//    //                result *= Int(pow(Double(factor), Double(power)/Double(n)))
//    //            }
//    //            factor += 1
//    //        }
//    //
//    //        if m > 1 {
//    //            return -1
//    //        }
//    //
//    //        // Cross-verify it
//    //        if Int(pow(Double(result), Double(n))) == m {
//    //            return result
//    //        }
//    //        return -1
//    //    }
//    
//    /*
//     Intuition Behind Binary Search for Nth Root
//     Why does binary search make sense?
//     
//     The answer, if it exists, is an integer X such that X^N = M.
//     
//     Any possible X must be in the range [1, M] (since X^N = M, and for N > 1, negative roots aren’t considered for this problem).
//     
//     As X increases, X^N grows monotonically (always increases), so the function is strictly increasing for X ≥ 1.
//     
//     Therefore, can apply binary search!
//     
//     If X^N < M, look to the right (bigger X).
//     
//     If X^N > M, look to the left (smaller X).
//     
//     If X^N == M, you found the answer.
//     
//     
//     */
//    
//    func nthRoot(_ m:Int, _ n:Int) -> Int {
//        var left = 1
//        var right = m
//        /*
//         Why not just use pow(Double, Double) in Swift?
//         Problems with pow(Double, Double) for integer Nth root search:
//         
//         Floating-point imprecision:
//         pow(Double, Double) returns a floating-point result, which can introduce rounding errors, especially for big numbers or higher roots. For example, pow(256.0, 0.25) should be 4, but due to floating-point imprecision, you might get 3.999999... or 4.0000001, and converting these to Int might cause the check if mid^N == M to fail.
//         
//         Overflow avoidance:
//         When working with large integers (for example, testing 64^6), directly computing using Int multiplication avoids intermediate overflows and floating-point inaccuracies. pow(Double, Double) might silently overflow or provide an incorrect result due to loss of precision.
//         */
//        func power(_ base: Int, _ e: Int) -> Int{
//            var result = 1
//            var e = e
//            while e>0 {
//                result *= base
//                if result > m { return m+1 } //early breakout
//                e -= 1
//            }
//            return result
//        }
//        
//        while left <= right {
//            let mid = (left + right) / 2
//            let val = power(mid, n)
//            if val == m {
//                return mid
//            } else if val < m {
//                left = mid + 1
//            } else {
//                right = mid - 1
//            }
//        }
//        
//        return -1
//    }
//    
//    
//    func medianOfRowSortedMatrix(_ matrix: [[Int]]) -> Int {
//        
//        // Since these matrix are row wise sorted and not globally sorted
//        /* Approach 1:
//         a. Flatten the matrix
//         b. Sort this matrix
//         c. Find the middle value
//         */
//        
//        /*
//         Approach 2:
//         a. We know that median is that value where all the elements on it's left are less than equal to it, here number of those elements will be ((n*m)+1)/2
//         b. left = min( all the values present on 0th column ), right = max( all the values present on last column), median will lie in between these values
//         c. For this mid, find number of elements present less than or equal to this value. Here go to each row, since each row is sorted use Binary Search to find the upperBound of this element in comparison to currMid value, add that into count
//         */
//        
//        let n = matrix.count
//        let m = matrix[0].count
//        let desiredCount = ((n*m)+1)/2
//        
//        //        func findNumberOfElementsLessThan(_ mid: Int) -> Int {
//        //            var count = 0
//        //            for row in matrix {
//        //                // Find the upperBound for this mid
//        //                var l = 0 , r = m-1
//        //                while l <= r {
//        //                    let m = (l+r)/2
//        //                    if row[m] <= mid {
//        //                        l = m+1
//        //                    } else {
//        //                        r = m-1
//        //                    }
//        //                }
//        //                count += l
//        //            }
//        //            return count
//        //        }
//        let findNumberOfElementsLessThan: (Int) -> Int = { mid in
//            var count = 0
//            for row in matrix {
//                // Find the upperBound for this mid
//                var l = 0 , r = m-1
//                while l <= r {
//                    let m = (l+r)/2
//                    if row[m] <= mid {
//                        l = m+1
//                    } else {
//                        r = m-1
//                    }
//                }
//                count += l
//            }
//            return count
//        }
//        
//        // left will be the minimum of all the elements present on 0th column of the matrix
//        var left = matrix.map { $0[0] }.min()!
//        var right = matrix.map { $0[m-1] }.max()!
//        
//        while left < right {
//            let mid = (left + right)/2
//            let val = findNumberOfElementsLessThan(mid)
//            if val == desiredCount {
//                return mid
//            } else if val < desiredCount {
//                left = mid + 1
//            } else {
//                right = mid
//            }
//        }
//        return left
//    }
//    
//    
//    // Can implement thinking of using higher order function like map
//    /*
//     // Higher-Order Function: Takes a matrix and a way to count elements <= mid
//     func medianOfMatrix(matrix: [[Int]], countFunc: (Int) -> Int) -> Int {
//     let n = matrix.count
//     let m = matrix[0].count
//     let desiredCount = (n * m + 1) / 2 // Median's position
//     
//     // Set initial search range
//     var left = matrix.map { $0 }.min()!
//     var right = matrix.map { $0[m-1] }.max()!
//     
//     while left < right {
//     let mid = (left + right) / 2
//     let count = countFunc(mid)
//     if count < desiredCount {
//     left = mid + 1
//     } else {
//     right = mid
//     }
//     }
//     return left
//     }
//     
//     // Example: Standard row-wise count function using a closure
//     let matrix = [
//     [1, 3, 8],
//     [2, 3, 4],
//     [1, 2, 5]
//     ]
//     
//     // Defines how to count number of elements <= `mid`
//     let rowCountFunc: (Int) -> Int = { mid in
//     var count = 0
//     for row in matrix {
//     var l = 0, r = row.count - 1
//     while l <= r {
//     let m = (l + r) / 2
//     if row[m] <= mid {
//     l = m + 1
//     } else {
//     r = m - 1
//     }
//     }
//     count += l
//     }
//     return count
//     }
//     
//     // Call the higher-order function with the closure
//     let median = medianOfMatrix(matrix: matrix, countFunc: rowCountFunc)
//     print(median) // Output: 3
//     */
//    
//    func singleNonDuplicate(_ nums: [Int]) -> Int {
//        // https://www.youtube.com/watch?v=PzszoiY5XMQ&t=15s
//        
//        /*
//         Approach 1 : XOR of all the numbers, a^a = 0 , a^0 = a , in the end you will get the unique element
//         Approach 2:
//         1,1,2,3,3,4,4
//         Answer = 2
//         
//         Left side of 2 observation
//         if we observe on left side of 2 we have pair (1,1) which have index 0,1 i.e. first element of the pair is on even index and second element of the pair is on odd index
//         
//         Right side of 2 observation
//         pair (3,3), first element of the pair is on odd index . Second element of the pair is on even index
//         
//         Xor of odd element with 1 => previous even 5 (101) ^ 1 => 100 i.e. 4
//         Xor of even element with 1 => next odd 4(100) ^ 1 => 101 i.e. 5
//         */
//        
//        var l = 0, r = nums.count - 2
//        while l < r {
//            //            let mid = (l + r) / 2
//            let mid = (l + r) >> 2
//            //check whether mid lies of left part of the answer or on right part of the answer
//            if nums[mid] == nums[mid^1] {
//                // I am on right side
//                l = mid + 1
//            } else {
//                r = mid - 1
//            }
//        }
//        return l
//    }
//    
//    //Searching in rotated sorted array
//    // Approach 1 : Using Index Mapping
//    /*
//     [4,5,6,7,8,0,1,2] target = 6
//     first find the pivot element i.e. 0 , it's index = 5 . 3 times left rotated ( using Binary Search in O (logn)
//     
//     realArray = [4,5,6,7,8,0,1,2]
//     
//     virtualSortedArray (VSA) = [0,1,2,4,5,6,7,8]
//     
//     indexFromRealArray = ( mid from VSA + pivot ) % n
//     example 1:
//     
//     indexFromRealArray = 6 = ( 1 (index of 1 in VSA) + 5 (pivot) ) % 8
//     
//     Now we will do the binary search on virtual Sorted array and will map to real Array index
//     */
//    //    func search(_ nums: [Int], _ target: Int) -> Int {
//    //
//    //        let n = nums.count
//    //
//    //        func findPivotIndex() -> Int {
//    //            // finding the index of smallest number
//    //            var l = 0, r = n - 1
//    //            while l < r {
//    //                let mid = (l + r) / 2
//    //                if nums[mid] > nums[r] {
//    //                    l = mid + 1
//    //                } else {
//    //                    r = mid
//    //                }
//    //            }
//    //            return l
//    //        }
//    //
//    //        let pivotIndex = findPivotIndex()
//    //
//    //        var l = 0, r = n - 1
//    //
//    //        while l <= r {
//    //            let midFromVirtualSortedArray = (l + r) / 2
//    //            let indexFromRealArray = (midFromVirtualSortedArray + pivotIndex) % n
//    //
//    //            if nums[indexFromRealArray] == target {
//    //                return indexFromRealArray
//    //            } else if nums[indexFromRealArray] < target {
//    //                l = midFromVirtualSortedArray + 1
//    //            } else {
//    //                r = midFromVirtualSortedArray - 1
//    //            }
//    //        }
//    //        return -1
//    //    }
//    // 2nd Approach :
//    /*
//     We know that [4,5,6,7,8,0,1,2] always atleast one half of the mid will always be sorted
//     
//     (4,5) is sorted
//     (6,7,8,0,1,2) is unsorted
//     
//     4,5,6,7,8,0 is unsorted
//     1,2 is sorted
//     
//     if nums[mid] > nums[low] i.e. left part is sorted and nums[mid] > target > nums[low] . Need to search in the left part of the array
//     */
//    func search(_ nums: [Int], _ target: Int) -> Int {
//        let n = nums.count
//        var l = 0, r = n - 1
//        while l < r {
//            let mid = (l + r) / 2
//            if nums[mid] == target {
//                return mid
//            } else if nums[l] <= nums[mid] {
//                // left part is sorted
//                if nums[l] <= target && target < nums[mid] {
//                    r = mid //search in left part
//                } else {
//                    l = mid + 1
//                }
//            } else if nums[mid] < nums[r] {
//                // right part is sorted
//                if nums[mid] < target && target <= nums[r] {
//                    l = mid + 1
//                } else {
//                    r = mid
//                }
//            }
//        }
//        return -1
//    }
//    
//    // This solution won't work if the elements are sparse and not dense
//    // [1,2,1e6]
//    // [4,5,1e2]
//    // Same for 2D matrix rowWise also if the elements are not dense, doing binary search won't work there
//    // func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
//    //     // quite similar to finding median in row sorted Matrix i.e. medianOfRowSortedMatrix
//    //     var left = min(nums1.first!, nums2.first!)
//    //     var right = max(nums1.last!, nums2.last!)
//    
//    //     let n = nums1.count
//    //     let m = nums2.count
//    
//    //     func countLesserThanOrEqual(_ middle: Double) -> Int {
//    //         var count = 0
//    //         var left = 0, right = n-1
//    //         while left < right {
//    //             let mid = (left+right)/2
//    //             if Double(nums1[mid]) <= middle {
//    //                 left = mid + 1
//    //             } else {
//    //                 right = mid
//    //             }
//    //         }
//    //         count += left
//    //         left = 0
//    //         right = m-1
//    //         while left < right {
//    //             let mid = (left+right)/2
//    //             if Double(nums2[mid]) <= middle {
//    //                 left = mid + 1
//    //             } else {
//    //                 right = mid
//    //             }
//    //         }
//    //         count += left
//    //         return count
//    //     }
//    
//    //     while left < right {
//    //         var middle = Double(left+right)/2
//    //         // Ideal median is such that there are exactly (n+m)/2 elements lesser than this middle value
//    //         var desiredValueCount = (n+m)/2
//    
//    //         let actualCount = countLesserThanOrEqual(middle)
//    //         if actualCount == desiredValueCount {
//    //             return middle
//    //         } else if actualCount < desiredValueCount {
//    //             left = Int(middle) + 1
//    //         } else {
//    //             right = Int(middle)
//    //         }
//    //     }
//    //     return Double(left)
//    // }
//    //    func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
//    //        let nums3 = (nums1+nums2).sorted()
//    //        let n = nums3.count
//    //        if n % 2 == 1 {
//    //           return Double(nums3[n/2])
//    //        } else {
//    //           return Double(nums3[n/2 - 1] + nums3[n/2]) / 2.0
//    //        }
//    //    }
//    // No need of storing the entire array, just need to know the n/2 and n/2 - 1 element
////    func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
////        var count = 0 // tracker for the index we are looking if incase we were filling new array
////        let n = nums1.count+nums2.count
////        var ind2 = n/2
////        var ind1 = ind2 - 1
////        var i = 0, j = 0
////        var ind1el = -1, ind2el = -1
////        
////        while i<nums1.count && j<nums2.count {
////            if nums1[i] < nums2[j] {
////                if count == ind2 { ind2el = nums1[i] }
////                if count == ind1 { ind1el = nums1[i] }
////                i += 1
////                count += 1
////            } else {
////                if count == ind2 { ind2el = nums2[j] }
////                if count == ind1 { ind1el = nums2[j] }
////                j += 1
////                count += 1
////            }
////        }
////        
////        while i<nums1.count {
////            if count == ind2 { ind2el = nums1[i] }
////            if count == ind1 { ind1el = nums1[i] }
////            i += 1
////            count += 1
////        }
////        
////        while j<nums2.count {
////            if count == ind2 { ind2el = nums2[j] }
////            if count == ind1 { ind1el = nums2[j] }
////            j += 1
////            count += 1
////        }
////        
////        if n%2 == 1 {
////            return Double(ind2el)
////        }
////        else {
////            return Double(ind1el+ind2el)/2.0
////        }
////    }
//    // Using Binary Search on Partition [IMP]: https://www.youtube.com/watch?v=F9c7LpRZWVQ
//    func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
//        let n1 = nums1.count , n2 = nums2.count, n = n1 + n2
//        if n1 > n2 { return findMedianSortedArrays(nums2, nums1) } // first array always the smaller array
//        var low = 0
//        var high = n1
//        
//        var left = (n1+n2+1)/2 // number of elements required on the left
//        
//        while low <= high {
//            var mid1 = (low+high)>>1
//            var mid2 = left - mid1
//            var l1 = Int.min, l2 = Int.min, r1 = Int.max, r2 = Int.max
//            
//            if mid1 < n1 { r1 = nums1[mid1] }
//            if mid2 < n2 { r2 = nums2[mid2] }
//            if mid1 - 1 >= 0 { l1 = nums1[mid1-1] }
//            if mid2 - 1 >= 0 { l2 = nums2[mid2-1] }
//            
//            if l1 <= r2 && l2 <= r1 {
//                // valid split
//                if n % 2 == 1 { return Double(max(l1, l2))}
//                else {
//                    return Double(max(l1, l2) + min(r1, r2)) / 2.0
//                }
//            } else if l1 > r2 {
//                high = mid1 - 1
//            } else {
//                low = mid1 + 1
//            }
//        }
//        return 0.0
//    }
//    
//    func findkThElementSortedArrays(_ nums1: [Int], _ nums2: [Int], _ k: Int) -> Int {
//        //This sounds like the similar approach to how we solved finding the median of two sorted arrays using binary search on the partition
//        // Example:
//        /*
//         nums1 = 1 2 5 7 9 11
//         nums2 = 3 4 6 8
//         k = 8
//         
//         nums1 + nums2 . sorted = 1 2 3 4 5 6 7 8 9 11
//         Answer = 8 (at 7th index)
//         
//         
//         I know that in combined sorted array there will be 7 elements on the left for this
//         let's take binary search over that
//         
//         l = 0 , r = 7
//         mid = 3
//         
//         I'm saying let's have 3 elements from nums1 and 7-3 elements from nums2 and check if it's the valid partition
//         
//          1 2 5| 7 9 11
//        3 4 6 8| Int_max
//         valid partition : NO : (8>7) l2>r1 thus try moving towards right
//         
//         right = mid+1 = 4
//         ...
//         
//         one solution we will get will be
//         1 2 5 7 | 9 11
//           3 4 6 | 8
//         valid partition: YES : 7<8 and 6<9 -> Answer is 8 (r2)
//         */
//        let n1 = nums1.count, n2 = nums2.count
//        if n1 > n2 { return findkThElementSortedArrays(nums2, nums1, k) } //Doing binary search on smallest array
//        
//        var left = max(0, k - n2), right = max(k, n1)
//        
//        while left <= right {
//            var mid = (left + right)>>1
//            
//            let cut1 = mid
//            let cut2 = k - mid
//            
//            // trying to take mid elements from nums1 and k-1-mid from nums2
//            var l1 = Int.min, l2 = Int.min, r1 = Int.max, r2 = Int.max
//            
//            l1 = (cut1 == 0) ? Int.min : nums1[cut1-1]
//            l2 = (cut2 == 0) ? Int.min : nums2[cut2-1]
//            r1 = (cut1 == n1) ? Int.max : nums1[cut1]
//            r2 = (cut2 == n2) ? Int.max : nums2[cut2]
//            
//            if l1 <= r2 && l2 <= r1 {
//                //valid cut
//                return max(l1,l2)
//            } else if l1 > r2 {
//                // Too many from nums1
//                right = mid - 1
//            } else {
//                left = mid + 1
//            }
//        }
//        return -1
//    }
//    
////    func bookAllocationRecursive(_ nums: [Int], _ k: Int) -> Int {
////        /*
////         This question I want to build and understand the concept, before jumping directly to the most optimized solution.
////         First for the basic approach I was thinking about using Recursion.
////         For m students we have m-1 partition. Now as per my understanding question boils down to find the max Sum of each of the elements residing in the partition to be the minimum, so that burden on each student is minimum given that we need to cover the entire syllabus i.e. all the books.
////         Now for this I was thinking about using recursion and exploring all the partitions, type of pseudoCode
////         maximumPagesAssignedToASingleStudent
////         func( nums, students, index, currSumPages ) {
////         if students == 0 && index == n {
////         return max(currSumPages, maximumPagesAssignedToASingleStudent)
////         }
////         2 cases either I can do the partition at index i by considering that into sum or I won't do the partition there, and since I want minimum of these so that burden on each student is minimum
////         min( func(nums, students, index+1, currSumPages), func(nums, students-1 , index+1, currSumPages+nums[index] )
////         }
////         Ofcourse there will be some tweaking in the pseudo code that needs to be done, but using it to illustrate my approach.
////         I want to build and understand this approach first, after that I want to explore how to use memoization over that, then the iterative DP solution for this.
////         After that in the end I will see if a standard optimal solution exist for this ( some sort of Greedy approach ).
////         */
////        let n = nums.count
////        func helper(_ index: Int, _ currentSum: Int, _ studentsLeft : Int) -> Int {
////            if index == n {
////                return studentsLeft == 1 ? currentSum : Int.max
////            }
////            
////            if studentsLeft == 0 {
////                return Int.max
////            }
////            
////            // Since assignment needs to be continous i.e. if students is assigned book 2, needs to have book 0,1 also
////            
////            let newSum = currentSum + nums[index]
////            
////            var result1 = helper(index+1, newSum, studentsLeft)
////
////            var result2 = Int.max
////            
////            if studentsLeft > 1 {
////                // here we are doing the partition
////                result2 = max(currentSum, helper(index+1, nums[index], studentsLeft-1))
////            }
////            return min(result1, result2)
////        }
////        return helper(0,0,k)
////    }
//    
//    // Understanding closure for recursive approach
////    func bookAllocationRecursive(_ nums: [Int], _ m: Int) -> Int {
////        
////        // Step 1: Declare the variable with explicit type
////        var closureHelper : (([Int], Int, Int, Int) -> Int)!
////        
////        // Step 2: Assign the closure implementation
////        closureHelper = { nums, index, studentsLeft, currentSum in
////            // Base case: processed all books
////            if index == nums.count {
////                return studentsLeft == 1 ? currentSum : Int.max
////            }
////            
////            if studentsLeft == 0 {
////                return Int.max
////            }
////            
////            let newSum = currentSum + nums[index]
////            
////            // Now closureHelper exists and can be called!
////            let continueWithCurrent = closureHelper(nums, index + 1, studentsLeft, newSum)
////            
////            let startNewStudent = studentsLeft > 1 ?
////                max(currentSum, closureHelper(nums, index + 1, studentsLeft - 1, nums[index])) :
////                Int.max
////            
////            return min(continueWithCurrent, startNewStudent)
////        }
////        
////        return closureHelper(nums, 0, m, 0)
////    }
////    
////        func bookAllocationMemo(_ nums: [Int], _ k: Int) -> Int {
////            let n = nums.count
////            // Thinking about having memoization of index, students
////            var memo : [String:Int] = [:]
////            
////            
////            func helper(_ index: Int, _ currentSum: Int, _ studentsLeft : Int) -> Int {
////                let key = "\(index)\(currentSum)\(studentsLeft)"
////                
////                if let cachedResult = memo[key] {
////                    return cachedResult
////                }
////                
////                if index == n {
////                    memo[key] = studentsLeft == 1 ? currentSum : Int.max
////                    return memo[key]!
////                }
////    
////                if studentsLeft == 0 {
////                    memo[key] = Int.max
////                    return Int.max
////                }
////    
////                // Since assignment needs to be continous i.e. if students is assigned book 2, needs to have book 0,1 also
////    
////                let newSum = currentSum + nums[index]
////    
////                var result1 = helper(index+1, newSum, studentsLeft)
////    
////                var result2 = Int.max
////    
////                if studentsLeft > 1 {
////                    // here we are doing the partition
////                    result2 = max(currentSum, helper(index+1, nums[index], studentsLeft-1))
////                }
////                memo[key] = min(result1, result2)
////                return min(result1, result2)
////            }
////            return helper(0,0,k)
////        }
//    
//    func bookAllocationDP(_ nums: [Int], _ m: Int) -> Int {
//        
//        let n = nums.count
//        if m > n {
//           return -1 //Number of students are greater than number of books
//        }
//        
//        // dp[i][j] : Minimum of maximum pages when allocating the first i books to j students
//        var dp = Array(repeating: Array(repeating: Int.max, count: m+1), count: n+1)
//        // our answer will be dp[k][n]
//        
//        // Calculate the prefix sum to keep it handy
//        
//        var prefixSumArray = Array(repeating: 0, count: n+1)
//        for i in 1...n {
//            prefixSumArray[i] = prefixSumArray[i-1] + nums[i-1]
//        }
//        
//        dp[0][0] = 0 // 0 books to 0 students : load 0
//        
//        for i in 1...n {
//            for j in 1...min(i,m){
//                // Observation since in memo: Using three variable in string we are using three loops here
//                // Check all the possible positions for last partition
//                // Taking decision for the last jth student
//                for k in j-1..<i { // j-1 books have been assigned that's the starting point for last partition,
//                    // Ensures j-1 students get at least 1 book each
//                    if dp[k][j-1] != Int.max {
//                        dp[i][j] = min(dp[i][j],max(dp[k][j-1], prefixSumArray[i]-prefixSumArray[k]))
//                    }
//                }
//            }
//        }
//        return dp[n][m]
//    }
//    
//    //https://www.youtube.com/watch?v=gYmWHvRHu-s&list=PLgUwDviBIf0p4ozDR_kJJkONnb1wdx2Ma&index=70
//    func bookAllocationBinarySearchOptimal(_ nums: [Int], _ m: Int) -> Int {
//        let n = nums.count
//        if m > n { return -1 }
//        
//        // Search space: [max(nums), sum(nums)]
//        let low = nums.max()!        // At least one student gets the largest book
//        let high = nums.reduce(0, +)  // One student gets all books
//        
//        var result = -1
//        var left = low, right = high
//        
//        var canAllocate : (Int) -> Bool = {
//            maxLoad in
//            var studentsUsed = 1
//            var currentLoad = 0
//            
//            for pages in nums {
//                if pages > maxLoad { return false }  // Single book exceeds limit
//                
//                if currentLoad + pages > maxLoad {
//                    // Start new student
//                    studentsUsed += 1
//                    currentLoad = pages
//                    
//                    if studentsUsed > m { return false }
//                } else {
//                    currentLoad += pages
//                }
//            }
//            
//            return studentsUsed <= m
//            
//        }
//        
//        // Binary search on the answer
//        while left <= right {  // Exact search pattern
//            let mid = (left + right) / 2
//            
//            if canAllocate(mid) {
//                result = mid          // Valid allocation found
//                right = mid - 1       // Try for smaller maximum
//            } else {
//                left = mid + 1        // Need larger maximum
//            }
//        }
//        
//        return result
//    }
//
//    func canAllocate(_ nums: [Int], _ m: Int, _ maxLoad: Int) -> Bool {
//        var studentsUsed = 1
//        var currentLoad = 0
//        
//        for pages in nums {
//            if pages > maxLoad { return false }  // Single book exceeds limit
//            
//            if currentLoad + pages > maxLoad {
//                // Start new student
//                studentsUsed += 1
//                currentLoad = pages
//                
//                if studentsUsed > m { return false }
//            } else {
//                currentLoad += pages
//            }
//        }
//        return studentsUsed <= m
//    }
//    
//    func agressiveCowsBinarySearchOptimal(_ nums: [Int], _ m: Int) -> Int {
//        let n = nums.count
//        if m > n { return -1 }
//        
//        let sortedNums = nums.sorted()
//        
//        var low = 1
//        var high = sortedNums[n-1]-sortedNums[0]
//        
//        func canAllocate(_ minDistance: Int) -> Bool {
//            var cowsPlaced = 1
//            var lastPosition = sortedNums[0]
//            for i in 1..<n {
//                if sortedNums[i] - lastPosition >= minDistance {
//                    // can accomodate
//                    cowsPlaced +=  1
//                    lastPosition = sortedNums[i]
//                    if cowsPlaced == m { return true }
//                }
//            }
//            return false
//        }
//        
//        var result = -1
//        
//        while low<=high {
//            var mid = (low+high)>>1
//            if canAllocate(mid) {
//                result = mid
//                low = mid + 1
//            } else {
//                high = mid - 1
//            }
//        }
//        
//        return result
//    }
//    
//    func aggressiveCowsRecursive(_ nums: [Int], _ k: Int) -> Int {
//        let sortedNums = nums.sorted()
//        var maxResult = -1
//        
//        // Generate all possible combinations of k positions
//        func generateCombinations(_ index: Int, _ currentCombination: [Int], _ remaining: Int) {
//            // Base case: selected k positions
//            if remaining == 0 {
//                let minDistance = calculateMinDistance(currentCombination)
//                print(currentCombination)
//                maxResult = max(maxResult, minDistance)
//                return
//            }
//            
//            // Base case: not enough positions left
//            if index >= sortedNums.count || sortedNums.count - index < remaining {
//                return
//            }
//            
//            // Include current position
//            generateCombinations(index + 1, currentCombination + [sortedNums[index]], remaining - 1)
//            
//            // Exclude current position
//            generateCombinations(index + 1, currentCombination, remaining)
//        }
//        
//        func calculateMinDistance(_ positions: [Int]) -> Int {
//            var minDist = Int.max
//            for i in 1..<positions.count {
//                minDist = min(minDist, positions[i] - positions[i-1])
//            }
//            return minDist
//        }
//        
//        generateCombinations(0, [], k)
//        return maxResult
//    }
//
//}
//
////let solution = Solution()
////print(solution.findPath([ [1, 0, 0, 0] , [1, 1, 0, 1], [1, 1, 0, 0], [0, 1, 1, 1] ]))
//
////let word = "Swapnil"
////let s = "qwertySwapnilqwerty"
////
////if let range = s.range(of: word) {
////    print("Found '\(word)' at range \(range)")     // prints: Found 'u' at range ...
////    print("Substring is:", s[range])               // prints: Substring is: u
////}
//
////let solution = Solution()
//////print(solution.medianOfRowSortedMatrix([ [1, 4, 9], [2, 5, 6], [3, 7, 8] ] ))
//////print(solution.search([4,5,6,7,8,0,1,2], 0))
//////print(solution.bookAllocationRecursive([12, 34, 67, 90], 2))
//////print(solution.bookAllocationDP([12, 34, 67, 90], 2))
////print(solution.bookAllocationBinarySearchOptimal([12, 34, 67, 90], 2))
////print(solution.agressiveCowsBinarySearchOptimal([0, 3, 4, 7, 10, 9], 4))
////print(solution.aggressiveCowsRecursive([0, 3, 4, 7, 10, 9], 4))
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
// //MARK: Uber preparation:
// 
//
////Implementing BFS and DFS traversal
//class Graph {
//    let adjacencyList: [Int: [Int]]
//    
//    init(adjacencyList: [Int: [Int]]) {
//        self.adjacencyList = adjacencyList
//    }
//    
//    // Standard BFS
//    func bfs(startVertex: Int) -> [Int] {
//        var visitedVertices: Set<Int> = []
//        var queue: [Int] = []
//        var order = [Int]()
//        
//        queue.append(startVertex)
//        visitedVertices.insert(startVertex)
//        
//        while !queue.isEmpty {
//            let vertex = queue.removeFirst()
//            order.append(vertex)
//            if let adjacentVertices = self.adjacencyList[vertex] {
//                for adjacentVertex in adjacentVertices {
//                    if !visitedVertices.contains(adjacentVertex) {
//                        queue.append(adjacentVertex)
//                        visitedVertices.insert(adjacentVertex)
//                    }
//                }
//            }
//        }
//        return order
//    }
//    
//    // Iterative DFS - order matches the recursive one by marking visited on pop, not push
//    func dfsIterative(startVertex: Int) -> [Int] {
//        var visitedVertices: Set<Int> = []
//        var stack: [Int] = []
//        var order: [Int] = []
//        
//        stack.append(startVertex)
//        // Do not mark as visited here
//        
//        while !stack.isEmpty {
//            let vertex = stack.removeLast()
//            if !visitedVertices.contains(vertex) {
//                visitedVertices.insert(vertex)
//                order.append(vertex)
//                if let adjacentVertices = self.adjacencyList[vertex] {
//                    // Use reversed to process in adjacency list order
//                    for adjacentVertex in adjacentVertices.reversed() {
//                        if !visitedVertices.contains(adjacentVertex) {
//                            stack.append(adjacentVertex)
//                        }
//                    }
//                }
//            }
//        }
//        return order
//    }
//    
//    // Recursive DFS
//    func dfsRecursive(startVertex: Int) -> [Int] {
//        var visitedVertices: Set<Int> = []
//        var order: [Int] = []
//        
//        func dfsHelper(vertex: Int) {
//            visitedVertices.insert(vertex)
//            order.append(vertex)
//            if let adjacentVertices = adjacencyList[vertex] {
//                for adjacentVertex in adjacentVertices {
//                    if !visitedVertices.contains(adjacentVertex) {
//                        dfsHelper(vertex: adjacentVertex)
//                    }
//                }
//            }
//        }
//        
//        dfsHelper(vertex: startVertex)
//        return order
//    }
//    
//    func numBusesToDestinationRecursive(_ routes: [[Int]], _ source: Int, _ target: Int) -> Int {
//        
//        var stopToBuses: [Int:[Int]] = [:] // adjacency list - Which bus will stop at which index
//        
//        for (busIndex,stops) in routes.enumerated() {
//            for stop in stops {
//                stopToBuses[stop, default: []].append(busIndex)
//                /*
//                 "If the dictionary already has a list for this stop, add the bus to that list."
//
//                 "If it does NOT have a list for this stop yet, first create an empty list, then add the bus to it."
//                 */
//            }
//        }
//        
//        if source == target {
//            return 0
//        }
//        
//        var minBuses = Int.max
//        var memo: [String:Int] = [:]
//        
//        func dfsHelper(currentStop: Int, busesTaken: Int, visitedBus: inout Set<Int>, visitedStop: inout Set<Int>) -> Int {
//            
//            let key = "\(currentStop)_\(busesTaken)_\(visitedBus.sorted())"
//            if let memoizedResult = memo[key] {
//                return memoizedResult
//            }
//            
//            if currentStop == target {
//                minBuses = min(busesTaken, minBuses)
//                return busesTaken
//            }
//            
//            // for each bus that can be taken from the current stop - explore it's path
//            guard let buses = stopToBuses[currentStop] else {
//                return Int.max
//            }
//            
//            var result = Int.max
//            for busIndex in buses {
//                // taking the bus if it has not been used in current path
//                if !visitedBus.contains(busIndex) {
//                    visitedBus.insert(busIndex)
//                    // Checking it's path
//                    for stop in routes[busIndex] {
//                        if !visitedStop.contains(stop) {
//                            visitedStop.insert(stop)
//                            let temp = dfsHelper(currentStop: stop, busesTaken: busesTaken+1, visitedBus: &visitedBus, visitedStop: &visitedStop)
//                            result = min(result,temp)
//                            visitedStop.remove(stop) //backtracking
//                        }
//                    }
//                    visitedBus.remove(busIndex) //backtracking
//                }
//            }
//            memo[key] = result
//            return result
//        }
//        
//        var visitedBuses: Set<Int> = []
//        var visitedStops: Set<Int> = []
//        
//        let answer =  dfsHelper(currentStop: source, busesTaken: 0, visitedBus: &visitedBuses, visitedStop: &visitedStops)
//        
//        return answer == Int.max ? -1 : answer
//    }
//    
//    func numBusesToDestination(_ routes: [[Int]], _ source: Int, _ target: Int) -> Int {
//       var stopToBuses : [Int:[Int]] = [:] // works as adjacency list
//       var visitedBuses = Set<Int>()
//       var visitedStops = Set<Int>()
//
//       for (busIndex,stops) in routes.enumerated() {
//            for stop in stops {
//                stopToBuses[stop, default:[]].append(busIndex) //If the stop doesn't exist in stopToBuses as key, first intialize that stop with empty array and then append busIndex
//            }
//       }
//
//       guard let validSource = stopToBuses[source] else {
//         if source == target {return 0}
//         else {return -1 }
//        }
//
//       var queue = [(Int,Int)]()
//       queue.append((source,0))
//
//       while !queue.isEmpty {
//            // Why do we need to have this size loop inside this , usually bfs doesn't require this - It's said usually to do the level order traversal and grouping the elements of level together
//            var size = queue.count //queue.count stores how many nodes are at the current level (i.e., at the current “distance” from the starting node).
//            while size != 0 {
//                var (currentStop, busesTakenSoFar) = queue.removeFirst()
//                visitedStops.insert(currentStop)
//                if currentStop == target { return busesTakenSoFar }
//                guard let busesExist = stopToBuses[currentStop] else { continue }
//                for busIndex in stopToBuses[currentStop]! {
//                    if !visitedBuses.contains(busIndex) {
//                        visitedBuses.insert(busIndex)
//                        // see which all routes do this bus goes
//                        for stop in routes[busIndex] {
//                            if !visitedStops.contains(stop){
//                                visitedStops.insert(stop)
//                                queue.append((stop,busesTakenSoFar+1))
//                            }
//                        }
//                    }
//                }
//                size -= 1
//            }
//       }
//       return -1
//    }
//    
//    func numIslands(_ grid: [[Character]]) -> Int {
//        // 1. DFS Approach
//        var visited = Array(repeating: Array(repeating: false, count: grid[0].count), count: grid.count)
//
//        var answer = 0
//        let m = grid.count
//        let n = grid[0].count
//
//        func isSafe(_ r: Int, _ c: Int) -> Bool {
//            if (r >= 0) && (c >= 0) && (r < m) && (c < n) && (grid[r][c] == "1") && !visited[r][c] { return true }
//            else {
//                return false
//            }
//        }
//
//        func dfsHelper(_ r: Int, _ c: Int) {
//            if !isSafe(r,c) {return}
//            visited[r][c] = true
//            dfsHelper(r+1,c)
//            dfsHelper(r,c+1)
//            dfsHelper(r,c-1)
//            dfsHelper(r-1,c)
//        }
//
//        for i in 0..<m {
//            for j in 0..<n {
//                if grid[i][j] == "0" { continue }
//                if !visited[i][j]{
//                    answer += 1 // I have found one island starting
//                    dfsHelper(i,j)
//                }
//            }
//        }
//
//       return answer
//    }
//    
//    func calcEquation(_ equations: [[String]], _ values: [Double], _ queries: [[String]]) -> [Double] {
//        var dsu = DSU()
//        // Building up the DSU
//        for (index, equation) in equations.enumerated() {
//            var firstChar = equation.first!, secondChar = equation.last!
//            dsu.union(firstChar, secondChar, values[index])
//        }
//        
//        var result = [Double]()
//        
//        for query in queries {
//            let x = query.first!, y = query.last!
//            let (rootX, weightXTillRoot) = dsu.find(x)
//            let (rootY, weightYTillRoot) = dsu.find(y)
//            
//            if rootX != rootY {
//                result.append(-1.00000)
//                continue
//            }
//            
//            result.append(weightXTillRoot/weightYTillRoot)
//        }
//        return result
//    }
//    
//}
//
//class DSU {
//    var parent = [String: String]()
//    var weight = [String: Double]()
//    
//    func add(_ x: String) {
//        parent[x] = x
//        weight[x] = 1.0
//    }
//    
//    func find(_ x: String) -> (String,Double) {
//        // return root of x and value of moving x till it's root
//        add(x)
//        if parent[x] != x {
//                let (root, parentWeight) = find(parent[x]!)
//                weight[x]! *= parentWeight
//                parent[x]! = root
//        }
//        return (parent[x]!, weight[x]!)
//    }
//    
//    func union(_ x: String, _ y: String, _ v : Double) {
//        add(x)
//        add(y)
//        
//        let (rootX, weightXTillRoot) = find(x)
//        let (rootY, weightYTillRoot) = find(y)
//        
//        if rootX != rootY {
//            parent[rootX] = rootY
//            weight[rootX] = v * weightXTillRoot/weightYTillRoot
//        }
//    }
//}
//
///*
// class Solution {
//     func numIslands(_ grid: [[Character]]) -> Int {
//         /*
//         // 1. DFS Approach
//         var visited = Array(repeating: Array(repeating: false, count: grid[0].count), count: grid.count)
//
//         var answer = 0
//         let m = grid.count
//         let n = grid[0].count
//
//         func isSafe(_ r: Int, _ c: Int) -> Bool {
//             if r < 0 || c < 0 || r >= m || c >= n { return false }
//             if grid[r][c] != "1" { return false }
//             if visited[r][c] { return false }
//             return true
//         }
//
//         func dfsHelper(_ r: Int, _ c: Int) {
//             if !isSafe(r,c) {return}
//             visited[r][c] = true
//             dfsHelper(r+1,c)
//             dfsHelper(r,c+1)
//             dfsHelper(r,c-1)
//             dfsHelper(r-1,c)
//         }
//
//         for i in 0..<m {
//             for j in 0..<n {
//                 if grid[i][j] == "0" { continue }
//                 /*
//                 if grid[r][c] == "1" {
//                         grid[r][c] = "0" // mark as visited
//                     }
//                 Use this approach if wanted space optimization
//                 */
//                 if !visited[i][j]{
//                     answer += 1 // I have found one island starting
//                     dfsHelper(i,j)
//                 }
//             }
//         }
//        return answer
//        */
//
//        // BFS Approach:
//         var visited = Array(repeating: Array(repeating: false, count: grid[0].count), count: grid.count)
//
//         var answer = 0
//         let m = grid.count
//         let n = grid[0].count
//
//         var queue : [[Int]] = [] // Queue conatains (i,j)
//
//         func isSafe(_ r: Int, _ c: Int) -> Bool {
//             if r < 0 || c < 0 || r >= m || c >= n { return false }
//             if grid[r][c] != "1" { return false }
//             if visited[r][c] { return false }
//             return true
//         }
//
//         func bfsTraversal() {
//             while !queue.isEmpty {
//                 var size = queue.count
//                 while size != 0 {
//                     var topElement = queue.removeFirst()
//                     var currRow = topElement.first!
//                     var currCol = topElement.last!
//                     visited[currRow][currCol] = true
//                     if isSafe(currRow, currCol+1) {
//                         visited[currRow][currCol+1] = true
//                         queue.append([currRow, currCol+1])
//                     }
//
//                     if isSafe(currRow+1, currCol) {
//                         visited[currRow+1][currCol] = true
//                         queue.append([currRow+1, currCol])
//                     }
//                     if isSafe(currRow-1, currCol) {
//                         visited[currRow-1][currCol] = true
//                         queue.append([currRow-1, currCol])
//                     }
//                     if isSafe(currRow, currCol-1) {
//                         visited[currRow][currCol-1] = true
//                         queue.append([currRow, currCol-1])
//                     }
//                     size -= 1
//                 }
//             }
//         }
//
//         for i in 0..<m {
//             for j in 0..<n {
//                 if isSafe(i,j) {
//                     queue.append([i,j])
//                     bfsTraversal()
//                     answer += 1
//                 }
//             }
//         }
//         return answer
//     }
//
//     /*
//     func numIslands(_ grid: [[Character]]) -> Int {
//     let m = grid.count
//     let n = grid[0].count
//     var visited = Array(repeating: Array(repeating: false, count: n), count: m)
//     var answer = 0
//
//     // All possible directions: right, down, left, up
//     let directions = [(0,1), (1,0), (0,-1), (-1,0)]
//
//     func isSafe(_ r: Int, _ c: Int) -> Bool {
//         return r >= 0 && c >= 0 && r < m && c < n
//             && grid[r][c] == "1"
//             && !visited[r][c]
//     }
//
//     func bfs(_ startR: Int, _ startC: Int) {
//         var queue = [[startR, startC]]
//         visited[startR][startC] = true
//         while !queue.isEmpty {
//             let (r, c) = (queue, queue[1])
//             queue.removeFirst()
//             for dir in directions {
//                 let nr = r + dir.0, nc = c + dir.1
//                 if isSafe(nr, nc) {
//                     visited[nr][nc] = true
//                     queue.append([nr, nc])
//                 }
//             }
//         }
//     }
//
//     for i in 0..<m {
//         for j in 0..<n {
//             if isSafe(i, j) {
//                 bfs(i, j)
//                 answer += 1
//             }
//         }
//     }
//     return answer
// }
//  Using row and col vector direction
//     */
// }
// */
//
//
//let adjacencyList = [
//    1: [2, 4],
//    2: [1, 3, 5, 8],
//    3: [2, 4, 9, 10],
//    4: [1, 3],
//    5: [2, 6, 7, 8],
//    6: [5],
//    7: [2, 5, 8],
//    8: [2, 5, 7],
//    9: [3],
//    10: [3]
//]
//
//let graph = Graph(adjacencyList: adjacencyList)
////print(graph.bfs(startVertex: 1))
////print(graph.dfsIterative(startVertex: 1))
////print(graph.dfsRecursive(startVertex: 1))
//
///*
// routes = [[1,2,7],[3,6,7]], source = 1, target = 6
// */
//
////print(graph.numBusesToDestination([[1,2,7],[3,6,7]], 1, 6))
//print(graph.numIslands([["1","1","1"],["0","1","0"],["1","1","1"]]))


/**
 * // This is the BinaryMatrix's API interface.
 * // You should not implement it, or speculate about its implementation
 * public protocol BinaryMatrix {
 *     func get(_ row: Int, _ col: Int) -> Int
 *     func dimensions() -> [Int]
 * }
 */

//protocol BinaryMatrix {
//    func get(_ row: Int, _ col: Int) -> Int
//    func dimensions() -> [Int]
//}
//
//class Solution {
//    func leftMostColumnWithOne(_ binaryMatrix: BinaryMatrix) -> Int {
//        let dims = binaryMatrix.dimensions()
//        let numRows = dims[0]
//        let numCols = dims[1]
//        
//        var ans = numCols
//        
//        // Your code here:
//        // Find the leftmost column with at least a 1
//        // You may use BinaryMatrix.get(row, col) and BinaryMatrix.dimensions()
//        // Be sure not to exceed 1000 calls to get()
//        
//        //Approach 1: Using Binary Search on each row to find the lowerBound of 1 and find the minimum of index acheived in the way : O (M logN )
//        /*
//         for row in 0..<numRows {
//         var l = 0 , r = numCols - 1
//         while l < r {
//         let mid = (l + r)>>1
//         if binaryMatrix.get(row,mid) == 1 {
//         r = mid
//         } else {
//         l = mid + 1
//         }
//         }
//         
//         // l could be possible 1
//         if binaryMatrix.get(row, l) == 1 {
//         ans = min(ans,l)
//         }
//         }
//         */
//        
//        //Approach 2 : O(M+N)
//        var row = 0 , col = numCols - 1
//        
//        func isSafe(_ r: Int, _ c: Int) -> Bool {
//            return r >= 0 && r<numRows && c>=0 && c<numCols
//        }
//        
//        while isSafe(row,col) {
//            if binaryMatrix.get(row,col) == 1 {
//                ans = min(ans, col)
//                col -= 1
//            } else {
//                row += 1
//            }
//        }
//        return ans == numCols ? -1 : ans
//    }
//    
//    func spiralOrder(_ matrix: [[Int]]) -> [Int] {
//        let m = matrix.count
//        let n = matrix[0].count
//
//        var result = [Int]()
//
//        func dfsHelper(_ top: Int, _ left: Int, _ right: Int, _ bottom: Int) {
//
//             // Base case
//             if left > right || top > bottom { return }
//
//             // Traverse all four directions for the boundaries
//             // Top row
//             for j in left...right {
//                result.append(matrix[top][j])
//             }
//
//             // Last column
//             for i in top...bottom {
//                result.append(matrix[i][right])
//             }
//
//             // Last row right to left
//             for j in stride(from: right, through: left, by:-1) {
//                result.append(matrix[bottom][j])
//             }
//
//             // First col from bottom to top
//             for i in stride(from: bottom, through: top, by: -1) {
//                result.append(matrix[i][left])
//             }
//
//             dfsHelper(top+1, left+1, right-1, bottom-1)
//        }
//        dfsHelper(0,0,n-1,m-1)
//        return result
//    }
//}
//
//
//class Dequeue<T> {
//    private var data = [T]()
//    
//    var isEmpty : Bool { data.isEmpty }
//    var count: Int { data.count }
//    
//    //append, prepend, popFront, popBack
//    func append(_ element: T){
//        data.append(element)
//    }
//    
//    func prepend(_ element: T){
//        data.insert(element, at: 0)
//    }
//    
//    @discardableResult
//    func popFront() -> T? {
//        return data.isEmpty ? nil : data.removeFirst()
//    }
//    
//    @discardableResult
//    func popBack() -> T? {
//        if !data.isEmpty {
//            return data.removeLast()
//        }
//        return nil
//    }
//    
//    func first() -> T? { data.first }
//    
//    func last() -> T? { data.last }
//}


//class Solution {
//    func fullJustify(_ words: [String], _ maxWidth: Int) -> [String] {
//        var output : [String] = []
//        var currWords : [String] = []
//        var currLength = 0
//        
//        var idx = 0
//        
//        //Helps in placing the white spaces evenly
//        func justifyHelper(_ isLastLine: Bool) {
//            // currWords = ["This", "is", "an"]
//            // currWordds.count = 3
//            // Totatl number of character
//            let characterCount = currWords.reduce(0) { partialResult, str in
//                partialResult + str.count
//            }
//            let totalSpacesRequired = maxWidth - characterCount
//            print("totatSpacesRequired for \(currWords) is \(totalSpacesRequired)")
//            var line = ""
//            if isLastLine || currWords.count == 1 {
//                // last line or single word
//                line = currWords.joined(separator: " ")
//                // line - "This "
//                // Adding the extra spaces now at end
//                let extra = maxWidth - line.count
//                if extra > 0 {
//                    line += String(repeating: " ", count: extra)
//                }
//            } else {
//                // example currWords = ["This", "is", "an"]
//                // line = "This is an" - number of white slots = 3 - 1 = 2
//                let numberOfWhiteSlots = currWords.count - 1
//                let evenSpaces = totalSpacesRequired / numberOfWhiteSlots
//                var oddExtraSpaces = totalSpacesRequired % numberOfWhiteSlots // These are basically extra spaces left which will go the left aligned space
//                
//                for j in 0..<currWords.count {
//                    line += currWords[j]
//                    if j != currWords.count - 1 {
//                        line += String(repeating: " ", count: evenSpaces + ((oddExtraSpaces > 0) ? 1 : 0) )
//                        if oddExtraSpaces > 0 {
//                            oddExtraSpaces -= 1
//                        }
//                    }
//                }
//            }
//            output.append(line)
//        }
//        
//        while idx<words.count {
//            let word = words[idx]
//            print("Current word \(word)")
//            if currWords.isEmpty {
//                if word.count <= maxWidth {
//                    currWords.append(word)
//                    currLength += word.count
//                    print("first if case : \(currWords)")
//                }
//            }
//            else if currLength + 1 + word.count <= maxWidth {
//                currWords.append(word)
//                currLength += 1 + word.count
//                print("second if case: \(currWords)")
//            } else {
//                // handle the current line and then switch to new line
//                print("third if case: \(currWords)")
//                justifyHelper(false)
//                currWords = [word]
//                currLength = word.count
//            }
//            idx += 1
//        }
//        justifyHelper(true)
//        return output
//    }
//}

//let solution = Solution()
//print(solution.fullJustify(["This", "is", "an", "example", "of", "text", "justification."], 16))


//class Solution {
////    func findWords(_ board: [[Character]], _ words: [String]) -> [String] {
////        var visited = Array(repeating: Array(repeating: false, count: board[0].count), count: board.count)
////        var output = Set<String>()
////        var wordSet = Set(words)
////        
////        let m = board.count
////        let n = board[0].count
////        
////        let directions = [(0,1),(1,0),(-1,0),(0,-1)]
////        
////        func dfsHelper(_ r: Int, _ c: Int, _ currWord: String){
////            if wordSet.contains(currWord) {
////                output.insert(currWord)
////            }
////            
////            for direction in directions {
////                let newR = r + direction.0
////                let newC = c + direction.1
////                if newR >= 0 && newC>=0 && newR<m && newC<n {
////                    if !visited[newR][newC]  {
////                        visited[newR][newC] = true
////                        dfsHelper(newR,newC, currWord+String(board[newR][newC]))
////                        visited[newR][newC] = false
////                    }
////                }
////            }
////        }
////        
////        for i in 0..<m {
////            for j in 0..<n {
////                if !visited[i][j] {
////                    visited[i][j] = true
////                    dfsHelper(i,j, String(board[i][j]))
////                    visited[i][j] = false
////                }
////            }
////        }
////        return Array(output)
////    }
//    
//    func findWords(_ board: [[Character]], _ words: [String]) -> [String] {
//        
//        //Build the trie data structure over words
//        let root = TrieNode()
//        for word in words {
//            var node = root
//            for wordCharacter in word {
//                if node.children[wordCharacter] == nil {
//                    node.children[wordCharacter] = TrieNode()
//                }
//                node = node.children[wordCharacter]!
//            }
//            node.word = word
//        }
//        
//        let directions = [(0,1),(1,0),(-1,0),(0,-1)]
//        let m = board.count
//        let n = board[0].count
//        var visited = Array(repeating: Array(repeating: false, count: n), count: m)
//        var output = Set<String>()
//        
//        func dfsHelper(_ r: Int, _ c: Int, _ node: TrieNode){
//            //Early pruning
//            let ch = board[r][c]
//            
//            guard let newNode = node.children[ch] else {
//                return
//            }
//            
//            if let foundWord = newNode.word {
//                output.insert(foundWord)
//            }
//            
//            for direction in directions {
//                let newR = r + direction.0
//                let newC = c + direction.1
//                if newR >= 0 && newC >= 0 && newR < m && newC < n {
//                    if !visited[newR][newC]{
//                        visited[newR][newC] = true
//                        dfsHelper(newR,newC,newNode)
//                        visited[newR][newC] = false
//                    }
//                }
//            }
//        }
//        
//        for i in 0..<m {
//            for j in 0..<n {
//                if !visited[i][j] {
//                    visited[i][j] = true
//                    dfsHelper(i,j,root)
//                    visited[i][j] = false
//                }
//            }
//        }
//        
//        return Array(output)
//    }
//}
//
//class TrieNode {
//    var children : [Character: TrieNode] = [:]
//    var word : String? = nil
//}

//class Solution {
//    
//    func buildAdjList(_ words:[String]) -> ([Character: Set<Character>]) {
//        var adjList = [Character:Set<Character>]()
//        let n = words.count
//        
//        for word in words {
//            for ch in word {
//                if adjList[ch] == nil {
//                    adjList[ch] = []
//                }
//            }
//        }
//        
//        for i in 0..<n-1{
//            let (first,second) = (words[i],words[i+1])
//            let minLength = min(first.count, second.count)
//            var found = false
//            for j in 0..<minLength {
//                let a = first[first.index(first.startIndex, offsetBy: j)] //Can solve using Array(firstWord)
//                let b = second[second.index(second.startIndex, offsetBy: j)]
//                if a != b {
//                    adjList[a, default: []].insert(b)
//                    found = true
//                    break
//                }
//            }
//            // abc , ab -> Invalid case, wrong order given to use
//            if !found && first.count > second.count {
//                return [:]
//            }
//        }
//        
//        return adjList
//    }
//    
//    func alienOrder(_ words: [String]) -> String {
//        var adjList = buildAdjList(words)
//        if adjList.isEmpty { return "" }
//        /*
//         1. Using DFS and external stack
//         */
//        var state: [Character:Int] = [:] //nil: unvisited, 1: visiting, 2: visited -> in directional graph we need three states to detect a cycle
//        var stackResult = [Character]()
//        var hasCycle =  false
//        
//        func dfs(_ ch: Character) {
//            if hasCycle { return }
//            
//            if let s = state[ch] {
//                if s==1 {
//                    hasCycle = true // found a cycle
//                }
//                return // possible values can 1,2 here i.e. visiting or visited
//            }
//            
//            state[ch] = 1
//            for neighbor in adjList[ch] ?? [] {
//                dfs(neighbor)
//                if hasCycle {return}
//            }
//            state[ch] = 2 //visited
//            stackResult.append(ch)
//        }
//        
//        for ch in adjList.keys {
//            if state[ch] == nil {
//                dfs(ch)
//                if hasCycle {
//                    return ""
//                }
//            }
//        }
//        return String(stackResult.reversed())
//    }
//    
//    func alienOrderUsingBFSKahnsAlgorithm(_ words:[String]) -> String {
//        var adjList = buildAdjList(words)
//        var indegree = [Character:Int]()
//        
//        // Initialize indegree for every character
//        for (ch, neighbors) in adjList {
//            if indegree[ch] == nil { indegree[ch] = 0 }
//            for neighbor in neighbors {
//                if indegree[neighbor] == nil { indegree[neighbor] = 0 }
//            }
//        }
//        
//        // Build indegree counts
//        for (ch, neighbors) in adjList {
//            for neighbor in neighbors {
//                indegree[neighbor]! += 1
//            }
//        }
//        
//        var queue = [Character]()
//        var output = [Character]()
//        
//        // Insert those chars which have indegree == 0
//        for (key,value) in indegree {
//            if value == 0 {
//                queue.append(key)
//            }
//        }
//        
//        while !queue.isEmpty {
//            var zeroInDegreeChar = queue.removeFirst()
//            output.append(zeroInDegreeChar)
//            
//            for neighbor in adjList[zeroInDegreeChar] ?? [] {
//                indegree[neighbor]! -= 1
//                if indegree[neighbor] == 0 {
//                    queue.append(neighbor)
//                }
//            }
//        }
//        
//        return output.count == indegree.count ? String(output) : ""
//    }
//    
//    func topoSort(_ adjList: [Int:[Int]], _ n: Int) -> [Int]{
//        /* 1. Topological sorting : https://www.youtube.com/watch?v=5lZ0iJMrUMk Using DFS, external stack and visited array, assuming Graph is DAG
//         If there's an edge from u to v , in order u should comes before v
//         */
//        /*
//        var visited = Array(repeating: false, count: adjList.count)
//        var stack = [Int]()
//        
//        func dfsHelper(_ node:Int){
//            visited[node] = true
//            for neighboor in adjList[node] ?? [] {
//                if !visited[neighboor]{
//                    dfsHelper(neighboor)
//                }
//            }
//            // Push back after visiting all the descendants (post order)
//            stack.append(node)
//        }
//        
//        for node in 0..<n {
//            if !visited[node]{
//                dfsHelper(node)
//            }
//        }
//        
//        // stack stores node stores the reverse topological order
//        return stack.reversed()
//         */
//        /*
//         2. Using BFS (Kahn's Algorithm) using indegree array: https://www.youtube.com/watch?v=73sneFXuTEg
//         Insert all the nodes with in degree == 0 in the queue, take them out and reduce the in degree of adjacent nodes by 1 , once their in degree becomes 0 insert them in the queue
//         How it works step-by-step (what your code does):
//         Compute in-degrees for all nodes.
//
//         Queue up all nodes that have in-degree 0 initially.
//
//         Pop nodes from the queue:
//
//         Add the node to your result list (that’s the next in order).
//
//         For each neighbor, decrement its in-degree.
//
//         If a neighbor's in-degree reaches 0, queue it up.
//
//         When the queue is empty, if you sorted all nodes, output contains a valid topological order.
//
//
//         */
//        var indegree = Array(repeating: 0, count: n)
//        for i in 0..<n {
//            for element in adjList[i] ?? []{
//                indegree[element] += 1
//            }
//        }
//        
//        var queue: [Int] = []
//        for i in 0..<n {
//            if indegree[i] == 0 {
//                queue.append(i)
//            }
//        }
//        
//        var output = [Int]()
//        while !queue.isEmpty {
//            var node = queue.removeFirst()
//            output.append(node)
//            // node is in your topo sort
//            // so please remove it from the indegree
//            for neighbor in adjList[node] ?? [] {
//                indegree[neighbor] -= 1
//                if indegree[neighbor] == 0 {
//                    queue.append(neighbor)
//                }
//            }
//        }
//        if output.count != n {
//            // Cycle detected (not a DAG)
//            return []
//        }
//        return output
//    }
//}

//class Solution {
////    func nearestPalindromic(_ n : String) -> String {
////        let length = n.count
////        let num = Int(n)!
////        var candidates = Set<Int>()
////        
////        // Edge palindromes (one more and one less digit)
////        candidates.insert(Int(pow(10, Double(length))) + 1)
////        if n.count > 1 {
////            candidates.insert(Int(pow(10, Double(length - 1))) - 1)
////        } else {
////            candidates.insert(0) // For single-digit n
////        }
////        
////        // Mirror, bump up, and bump down from middle
////        addMirroredValuePalindrome(num, &candidates, length)
////        addIncreasedMiddleValuePalindrome(num, &candidates)
////        addDecreasedMiddleValuePalindrome(num, &candidates)
////        
////        // Remove itself
////        candidates.remove(num)
////        
////        // Pick closest with rules
////        var result = -1
////        for c in candidates {
////            if result == -1 ||
////                abs(c - num) < abs(result - num) ||
////                (abs(c - num) == abs(result - num) && c < result) {
////                result = c
////            }
////        }
////        return String(result)
////    }
////    // ...helpers as above...
////    func addMirroredValuePalindrome(_ num: Int, _ candidates: inout Set<Int>, _ length: Int) {
////        let s = String(num)
////        let chars = Array(s)
////        let halfLen = (length + 1) / 2
////        var half = String(chars.prefix(halfLen))
////        
////        // Mirror function: create a palindrome by copying left to right
////        var mirrored: [Character]
////        if length % 2 == 0 {
////            // Even length, mirror full half to right (e.g., "12" => "1221")
////            mirrored = Array(half + String(half.reversed()))
////        } else {
////            // Odd length, mirror all but central char (e.g., "123" => "121")
////            mirrored = Array(half + String(half.dropLast().reversed()))
////        }
////        
////        let palindromeNum = Int(String(mirrored))!
////        candidates.insert(palindromeNum)
////    }
////
////    
////    func addIncreasedMiddleValuePalindrome(_ num: Int, _ candidates: inout Set<Int>) {
////        let s = String(num)
////        var chars = Array(s)
////        let halfLen = (chars.count + 1) / 2
////        var halfNum = Int(String(chars.prefix(halfLen)))! + 1
////        let half = String(halfNum)
////        var mirrored: [Character]
////        if chars.count % 2 == 0 {
////            mirrored = Array(half + String(half.reversed()))
////        } else {
////            mirrored = Array(half + String(half.dropLast().reversed()))
////        }
////        if let palindromeNum = Int(String(mirrored)) {
////            candidates.insert(palindromeNum)
////        }
////    }
////
////    func addDecreasedMiddleValuePalindrome(_ num: Int, _ candidates: inout Set<Int>) {
////        let s = String(num)
////        var chars = Array(s)
////        let halfLen = (chars.count + 1) / 2
////        var halfNum = Int(String(chars.prefix(halfLen)))! - 1
////        let half = String(halfNum)
////        // Handle underflow: e.g., 100.. -> 99.., pad zeros to left if needed
////        var halfPad = half
////        while halfPad.count < halfLen {
////            halfPad = "0" + halfPad
////        }
////        var mirrored: [Character]
////        if chars.count % 2 == 0 {
////            mirrored = Array(halfPad + String(halfPad.reversed()))
////        } else {
////            mirrored = Array(halfPad + String(halfPad.dropLast().reversed()))
////        }
////        if let palindromeNum = Int(String(mirrored)) {
////            candidates.insert(palindromeNum)
////        }
////    }
//
//    func nearestPalindromic(_ n: String) -> String {
//        let number = Int(n)!
//        let length = n.count
//        
//        var candidates = Set<Int>()
//        
//        //Cases like 99 -> 101
//        var nextSmallestPalindromeWithOneExtraDigit = Int(powl(10,Double(length)) + 1)
//        candidates.insert(nextSmallestPalindromeWithOneExtraDigit)
//        
//        //Cases like 101,102 -> 99 = 10^(len-1)-1
//        if length>1 {
//            var prevBiggestPalindromeWithOneLessDigit = Int(powl(10, Double(length-1)-1))
//            candidates.insert(prevBiggestPalindromeWithOneLessDigit)
//        } else {
//            candidates.insert(0)
//        }
//        
//        addMirrorHalf(length, number, &candidates)
//        addOneFromMiddle(length, number, &candidates)
//        decreaseOneFromMiddle(length, number, &candidates)
//        
//        var ans = -1
//        for candidate in candidates {
//            if abs(candidate-number) < abs(ans-number) ||
//                ans == -1 ||
//                (abs(candidate-number) == abs(ans-number) && candidate < ans ) {
//                ans = candidate
//            }
//        }
//        return String(ans)
//    }
//    
//    func addMirrorHalf(_ len: Int, _ number: Int, _ candidates: inout Set<Int>) {
//        let s = String(number)
//        var stringArray = Array(s)
//        var halfLength = (len+1)/2
//        var output: [Character]
//        var half = String(stringArray.prefix(halfLength))
//        
//        //1234 -> 12 + 21 -> 1221
//        if len % 2 == 0 {
//            output = Array(half+half.reversed())
//        } else {
//        //12345 -> 123 -> 12 + 3 + 21
//            output = Array(half + half.dropLast().reversed())
//        }
//        let answer = Int(String(output))!
//        candidates.insert(answer)
//    }
//    
//    func addOneFromMiddle(_ len: Int, _ number: Int, _ candidates: inout Set<Int>){
//        //12345 -> 123 -> 12 + (3->4) + 21 -> 12421
//        //1234 -> 1 + 23 + 4 -> 1 + 33 + 4 -> 1334
//        let s = String(number)
//        var stringArray = Array(s)
//        var halfLength = (len+1)/2
//        var output: [Character]
//        var half = String(stringArray.prefix(halfLength))
//        var halfNumber = Int(half)! + 1
//        half = String(halfNumber)
//        
//        if len % 2 == 0 {
//            output = Array(half + half.reversed())
//        } else {
//            output = Array(half + half.dropLast().reversed())
//        }
//        
//        let answer = Int(String(output))!
//        candidates.insert(answer)
//    }
//    
//    func decreaseOneFromMiddle(_ len: Int, _ number: Int, _ candidates: inout Set<Int>){
//                let s = String(number)
//                var chars = Array(s)
//                let halfLen = (chars.count + 1) / 2
//                var halfNum = Int(String(chars.prefix(halfLen)))! - 1
//                let half = String(halfNum)
//                // Handle underflow: e.g., 100.. -> 99.., pad zeros to left if needed
//                var halfPad = half
//                print("decrement half: \(half)")
//                while halfPad.count < halfLen {
//                    halfPad = "0" + halfPad
//                }
//                print("decrement half pad: \(halfPad)")
//                var mirrored: [Character]
//                if chars.count % 2 == 0 {
//                    mirrored = Array(halfPad + String(halfPad.reversed()))
//                } else {
//                    mirrored = Array(halfPad + String(halfPad.dropLast().reversed()))
//                }
//                if let palindromeNum = Int(String(mirrored)) {
//                    print("decrement palindromeNum made: \(palindromeNum)")
//                    candidates.insert(palindromeNum)
//                }
//    }
//}
//
//let solution = Solution()
//print(solution.nearestPalindromic("100"))


class Solution {
    func nearestPalindromic(_ n: String) -> String {
        let num = Int(n)!
        let len = n.count
        var candidates = Set<Int>()
        
        // 99->101, 100->99
        candidates.insert(Int(pow(10, Double(len))+1))
        candidates.insert(Int(pow(10, Double(len-1))-1))
        
        let halfLen = (len+1)/2
        var prefix = Int(n.prefix(halfLen))!
        
        //Middle element manipulation -1, 0(just mirroring), +1
        for i in -1...1 {
            // 1234 -> 1221(pure mirror), 1331, 1111
            // 12345 -> 12321, 12421, 12121
            
            var half = String(prefix + i)
            print("half \(half)")
            
            //padding with zerores if needed -> 10
            /*
             Need to handle that change
             */
            while half.count < halfLen {
                half = "0" + half
            }
            
            var palindrome: String
            
            if len % 2 == 0 {
                palindrome = half + String(half.reversed())
            }
            else {
                palindrome = half + String(half.dropLast().reversed())
            }
            
            if let palNum = Int(palindrome) {
                candidates.insert(palNum)
            }
        }
        
        candidates.remove(num)
        
        var answer = -1
        for candidate in candidates {
            if answer == -1 ||
                abs(candidate-num) < abs(answer-num) ||
                (abs(candidate-num) == abs(answer-num) && candidate<answer)
            {
                answer = candidate
            }
        }
        return String(answer)
    }
}

let solution = Solution()
print(solution.nearestPalindromic("10"))
