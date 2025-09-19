//import Foundation
//////
//////struct IntPair : Hashable {
//////    let x : Int
//////    let y : Int
//////}
//////
//////class Solution {
//////    func maxSumCombinations(_ nums1: [Int], _ nums2: [Int], _ k : Int) -> [Int]{
//////
//////        var visited :  Set<IntPair> = []
//////        var maxHeap = MaxHeap()
//////        
//////        let sortedNums1 = nums1.sorted(by: >)
//////        let sortedNums2 = nums2.sorted(by: >)
//////        
//////        let n1 = sortedNums1.count
//////        let n2 = sortedNums2.count
//////        
//////        var output = [Int]()
//////        
//////        maxHeap.insert([sortedNums1[0]+sortedNums2[0],0,0])  //contains sum, i, j
//////        visited.insert(IntPair(x: 0, y: 0))
//////        
//////        for operation in 0..<k {
//////            guard let maxElementInfo = maxHeap.extractMax() else {
//////                break
//////            }
//////            var currBestSum = maxElementInfo[0]
//////            var currI = maxElementInfo[1]
//////            var currJ = maxElementInfo[2]
//////            
//////            // Next Possible combinations is i,j+1 or i+1,j
//////            print("currBestSum \(currBestSum) currI \(currI) currJ \(currJ)")
//////            
//////            output.append(currBestSum)
//////            
//////            if currI + 1 < n1 {
//////                if !visited.contains(IntPair(x: currI + 1, y: currJ)){
//////                    visited.insert(IntPair(x: currI + 1, y: currJ))
//////                    maxHeap.insert([sortedNums1[currI+1]+sortedNums2[currJ],currI+1,currJ])
//////                }
//////            }
//////            
//////            if currJ + 1 < n2 {
//////                if !visited.contains(IntPair(x: currI, y: currJ+1)){
//////                    visited.insert(IntPair(x: currI, y: currJ+1))
//////                    maxHeap.insert([sortedNums1[currI]+sortedNums2[currJ+1],currI,currJ+1])
//////                }
//////            }
//////        }
//////        
//////        return output
//////    }
//////    
//////    func mergeKSortedArrays(_ nums:[[Int]], _ k: Int) -> [Int] {
//////        var output = [Int]()
//////        var minHeap = MinHeap()
//////        
//////        for i in 0..<k {
//////            //insert the first element of K sorted arrays in minHeap
//////            minHeap.insert((nums[i][0],i,0))
//////        }
//////        
//////        while !minHeap.isEmpty {
//////            guard  var (currMinValue, currMinRowIdx, currMinIdx) = minHeap.extractMin() else {
//////                break
//////            }
//////            output.append(currMinValue)
//////            if currMinIdx + 1 < nums[currMinRowIdx].count {
//////                minHeap.insert((nums[currMinRowIdx][currMinIdx+1], currMinRowIdx, currMinIdx + 1))
//////            }
//////        }
//////        
//////        return output
//////    }
//////}
//////
//////class MinHeap {
//////    var minHeap = [(Int,Int,Int)]() // This will contain the element and index like from which [[nums]] row this element exist from and in that which index I am on
//////    
//////    var isEmpty : Bool {
//////        minHeap.isEmpty
//////    }
//////    
//////    var count : Int {
//////        minHeap.count
//////    }
//////    
//////    var peek : Int? {
//////        minHeap.first?.0
//////    }
//////    
//////    //Heapify up
//////    func insert(_ x: (Int,Int,Int)) {
//////        minHeap.append(x)
//////        var idx = count - 1
//////        
//////        while idx > 0 {
//////            var parentIdx = (idx - 1)/2
//////            if minHeap[parentIdx].0 > minHeap[idx].0 {
//////                minHeap.swapAt(parentIdx, idx)
//////                idx = parentIdx
//////            } else {
//////                break
//////            }
//////        }
//////    }
//////    
//////    //Heapify Down
//////    func extractMin() -> (Int,Int,Int)? {
//////        guard !isEmpty else {
//////            return nil
//////        }
//////        
//////        var answer = minHeap[0]
//////        
//////        if minHeap.count > 1 {
//////            minHeap[0] = minHeap.removeLast()
//////            var parentIdx = 0
//////            while true {
//////                var leftChildIdx = 2 * parentIdx + 1
//////                var rightChildIdx = 2 * parentIdx + 2
//////                
//////                var smallestIdx = parentIdx
//////                
//////                if leftChildIdx < count && minHeap[smallestIdx].0 > minHeap[leftChildIdx].0 {
//////                    smallestIdx = leftChildIdx
//////                }
//////                
//////                if rightChildIdx < count && minHeap[smallestIdx].0 > minHeap[rightChildIdx].0 {
//////                    smallestIdx = rightChildIdx
//////                }
//////                
//////                if parentIdx == smallestIdx {
//////                    break
//////                }
//////                
//////                minHeap.swapAt(smallestIdx, parentIdx)
//////                parentIdx = smallestIdx
//////            }
//////        } else {
//////            minHeap.removeLast()
//////        }
//////        return answer
//////    }
//////}
//////
//////class MaxHeap {
//////    var maxHeap = [[Int]]()
//////    
//////    var isEmpty : Bool {
//////        maxHeap.isEmpty
//////    }
//////    
//////    var count: Int{
//////        maxHeap.count
//////    }
//////    
//////    var peek: Int? {
//////        maxHeap.first?.first
//////    }
//////    
//////    //Heapify up
//////    func insert(_ x: [Int]) {
//////        maxHeap.append(x)
//////        var idx = maxHeap.count - 1
//////        while idx > 0 {
//////            var parentIdx = (idx - 1)/2
//////            if maxHeap[parentIdx][0] < maxHeap[idx][0] {
//////                maxHeap.swapAt(parentIdx, idx)
//////                idx = parentIdx
//////            } else {
//////                break
//////            }
//////        }
//////    }
//////    
//////    //Heapify Down
//////    func extractMax() -> [Int]? {
//////        guard !maxHeap.isEmpty else { return nil }
//////        var answer = maxHeap[0]
//////        
//////        // If there's more than one element, move the last to the front and heapify down.
//////        if maxHeap.count > 1 {
//////            maxHeap[0] = maxHeap.removeLast()
//////            var parentIdx = 0
//////            
//////            while true {
//////                let leftChildIdx = 2 * parentIdx + 1
//////                let rightChildIdx = 2 * parentIdx + 2
//////                var largestIdx = parentIdx
//////                
//////                if leftChildIdx < maxHeap.count && maxHeap[leftChildIdx][0] > maxHeap[largestIdx][0] {
//////                    largestIdx = leftChildIdx
//////                }
//////                
//////                if rightChildIdx < maxHeap.count && maxHeap[rightChildIdx][0] > maxHeap[largestIdx][0] {
//////                    largestIdx = rightChildIdx
//////                }
//////                
//////                if largestIdx == parentIdx {
//////                    break
//////                }
//////                
//////                maxHeap.swapAt(parentIdx, largestIdx)
//////                parentIdx = largestIdx
//////            }
//////        } else {
//////            // If it's the last element, just remove it.
//////            maxHeap.removeLast()
//////        }
//////        
//////        return answer
//////    }
//////}
//////
//////let solution = Solution()
////////print(solution.maxSumCombinations([10,2,1], [10,5,4,1], 6))
//////print(solution.mergeKSortedArrays([[3,6,9], [2,4,6,8], [1,3,5,7]], 3))
////
////
////
////
////class MedianFinder {
////    var minHeap : MinHeap
////    var maxHeap : MaxHeap
////
////    init() {
////        minHeap = MinHeap()
////        maxHeap = MaxHeap()
////    }
////    
////    func balance() {
////        let minCount = minHeap.count
////        let maxCount = maxHeap.count
////
////        if maxCount - minCount > 1 {
////            // Move the root of maxHeap to minHeap
////            if let maxVal = maxHeap.extractMax() {
////                minHeap.insert(maxVal)
////            }
////        } else if minCount - maxCount > 1 {
////            // Move the root of minHeap to maxHeap
////            if let minVal = minHeap.extractMin() {
////                maxHeap.insert(minVal)
////            }
////        }
////    }
////    
////    /*
////    Insert Logic:
////
////    Smaller element → maxHeap
////    Larger element → minHeap
////    Balance Logic:
////
////    If one heap has more than one extra element compared to the other, transfer the top element from the larger heap to the smaller heap.
////    Median Calculation Logic:
////
////    If heaps are balanced → Median = Mean of tops of both heaps.
////    If heaps are imbalanced → Median = Top of the larger heap.
////    */
////
//////
//////    func addNum(_ num: Int) {
//////    if maxHeap.isEmpty {
//////        maxHeap.insert(num)
//////        return
//////    }
//////
//////    // Compare with the top of maxHeap
//////    if let maxSmallestElement = maxHeap.peek, num <= maxSmallestElement {
//////        maxHeap.insert(num)
//////    } else if let minLargestElement = minHeap.peek, num >= minLargestElement {
//////        // Compare with the top of minHeap
//////        minHeap.insert(num)
//////    } else {
//////        // Falls in between maxHeap and minHeap tops
//////        maxHeap.insert(num)
//////    }
//////
//////    // Balance the heaps after insertion
//////    balance()
//////    }
//////    
//////    func findMedian() -> Double {
//////        // if heaps are balanced
//////        if minHeap.count == maxHeap.count {
//////            return Double(Double(minHeap.peek!+maxHeap.peek!)/2.0)
//////        }
//////        if minHeap.count > maxHeap.count {
//////            return Double(minHeap.peek!)
//////        }
//////        return Double(maxHeap.peek!)
//////    }
//////}
//////
//////
//////class MinHeap {
//////    var minHeap = [Int]()
//////    
//////    var isEmpty : Bool {
//////        minHeap.isEmpty
//////    }
//////    
//////    var count : Int {
//////        minHeap.count
//////    }
//////    
//////    var peek : Int? {
//////        minHeap.first
//////    }
//////    
//////    //Heapify up
//////    func insert(_ x: Int) {
//////        minHeap.append(x)
//////        var idx = count - 1
//////        
//////        while idx > 0 {
//////            var parentIdx = (idx - 1)/2
//////            if minHeap[parentIdx] > minHeap[idx] {
//////                minHeap.swapAt(parentIdx, idx)
//////                idx = parentIdx
//////            } else {
//////                break
//////            }
//////        }
//////    }
//////    
//////    //Heapify Down
//////    func extractMin() -> Int? {
//////        guard !isEmpty else {
//////            return nil
//////        }
//////        
//////        var answer = minHeap[0]
//////        
//////        if minHeap.count > 1 {
//////            minHeap[0] = minHeap.removeLast()
//////            var parentIdx = 0
//////            while true {
//////                var leftChildIdx = 2 * parentIdx + 1
//////                var rightChildIdx = 2 * parentIdx + 2
//////                
//////                var smallestIdx = parentIdx
//////                
//////                if leftChildIdx < count && minHeap[smallestIdx] > minHeap[leftChildIdx] {
//////                    smallestIdx = leftChildIdx
//////                }
//////                
//////                if rightChildIdx < count && minHeap[smallestIdx] > minHeap[rightChildIdx] {
//////                    smallestIdx = rightChildIdx
//////                }
//////                
//////                if parentIdx == smallestIdx {
//////                    break
//////                }
//////                
//////                minHeap.swapAt(smallestIdx, parentIdx)
//////                parentIdx = smallestIdx
//////            }
//////        } else {
//////            minHeap.removeLast()
//////        }
//////        return answer
//////    }
//////}
//////
//////
//////class MaxHeap {
//////    var maxHeap = [Int]()
//////    
//////    var isEmpty : Bool {
//////        maxHeap.isEmpty
//////    }
//////    
//////    var count: Int{
//////        maxHeap.count
//////    }
//////    
//////    var peek: Int? {
//////        maxHeap.first
//////    }
//////    
//////    //Heapify up
//////    func insert(_ x: Int) {
//////        maxHeap.append(x)
//////        var idx = maxHeap.count - 1
//////        while idx > 0 {
//////            var parentIdx = (idx - 1)/2
//////            if maxHeap[parentIdx] < maxHeap[idx] {
//////                maxHeap.swapAt(parentIdx, idx)
//////                idx = parentIdx
//////            } else {
//////                break
//////            }
//////        }
//////    }
//////    
//////    //Heapify Down
//////    func extractMax() -> Int? {
//////        guard !maxHeap.isEmpty else { return nil }
//////        var answer = maxHeap[0]
//////        
//////        // If there's more than one element, move the last to the front and heapify down.
//////        if maxHeap.count > 1 {
//////            maxHeap[0] = maxHeap.removeLast()
//////            var parentIdx = 0
//////            
//////            while true {
//////                let leftChildIdx = 2 * parentIdx + 1
//////                let rightChildIdx = 2 * parentIdx + 2
//////                var largestIdx = parentIdx
//////                
//////                if leftChildIdx < maxHeap.count && maxHeap[leftChildIdx] > maxHeap[largestIdx] {
//////                    largestIdx = leftChildIdx
//////                }
//////                
//////                if rightChildIdx < maxHeap.count && maxHeap[rightChildIdx] > maxHeap[largestIdx] {
//////                    largestIdx = rightChildIdx
//////                }
//////                
//////                if largestIdx == parentIdx {
//////                    break
//////                }
//////                
//////                maxHeap.swapAt(parentIdx, largestIdx)
//////                parentIdx = largestIdx
//////            }
//////        } else {
//////            // If it's the last element, just remove it.
//////            maxHeap.removeLast()
//////        }
//////        return answer
//////    }
//////}
////
////
/////**
//// * Your MedianFinder object will be instantiated and called as such:
//// * let obj = MedianFinder()
//// * obj.addNum(num)
//// * let ret_2: Double = obj.findMedian()
//// */
////
//// /*
//// Optimization using Binary Search if element [0,100]
//// class MedianFinder {
////    var frequency = [Int](repeating: 0, count: 101) // Frequency array for range [0, 100]
////    var totalCount = 0                              // Total number of elements added
////
////    init() {}
////
////    func addNum(_ num: Int) {
////        frequency[num] += 1  // Increment frequency for the number
////        totalCount += 1      // Increment total count of numbers
////    }
////
////    func findMedian() -> Double {
////        let targetLeft = (totalCount + 1) / 2       // Target index for the left median (for odd/even cases)
////        var cumulativeCount = 0                    // Tracks the cumulative frequency
////
////        var leftMedian = -1
////        var rightMedian = -1
////
////        for i in 0...100 {
////            cumulativeCount += frequency[i]        // Update cumulative count
////
////            if cumulativeCount >= targetLeft && leftMedian == -1 {
////                leftMedian = i                     // First median (left side for even, or middle for odd)
////            }
////            if cumulativeCount >= (totalCount / 2 + 1) {
////                rightMedian = i                    // Second median for even-sized streams
////                break
////            }
////        }
////
////        if totalCount % 2 == 0 {
////            return Double(leftMedian + rightMedian) / 2.0 // Average for even-sized streams
////        } else {
////            return Double(leftMedian)                     // Middle for odd-sized streams
////        }
////    }
////}
////*/
//class MinHeap {
//    var minHeap = [Int]() // This will contain the element and index like from which [[nums]] row this element exist from and in that which index I am on
//
//    var isEmpty : Bool {
//        minHeap.isEmpty
//    }
//
//    var count : Int {
//        minHeap.count
//    }
//
//    var peek : Int? {
//        minHeap.first
//    }
//
//    //Heapify up
//    func insert(_ x: Int) {
//        minHeap.append(x)
//        var idx = count - 1
//
//        while idx > 0 {
//            var parentIdx = (idx - 1)/2
//            if minHeap[parentIdx] > minHeap[idx] {
//                minHeap.swapAt(parentIdx, idx)
//                idx = parentIdx
//            } else {
//                break
//            }
//        }
//    }
//
//    //Heapify Down
//    func extractMin() -> Int? {
//        guard !isEmpty else {
//            return nil
//        }
//
//        var answer = minHeap[0]
//
//        if minHeap.count > 1 {
//            minHeap[0] = minHeap.removeLast()
//            var parentIdx = 0
//            while true {
//                var leftChildIdx = 2 * parentIdx + 1
//                var rightChildIdx = 2 * parentIdx + 2
//
//                var smallestIdx = parentIdx
//
//                if leftChildIdx < count && minHeap[smallestIdx] > minHeap[leftChildIdx] {
//                    smallestIdx = leftChildIdx
//                }
//
//                if rightChildIdx < count && minHeap[smallestIdx] > minHeap[rightChildIdx] {
//                    smallestIdx = rightChildIdx
//                }
//
//                if parentIdx == smallestIdx {
//                    break
//                }
//
//                minHeap.swapAt(smallestIdx, parentIdx)
//                parentIdx = smallestIdx
//            }
//        } else {
//            minHeap.removeLast()
//        }
//        return answer
//    }
//
//}
//class Solution {
//    func maxEvents(_ events: [[Int]]) -> Int {
//        /*
//        var sortedEvents = events.sorted {
//            if $0.last! == $1.last! {
//                return $0.first! < $1.first!
//            }
//            return $0.last! < $1.last!
//        }
//        print(sortedEvents)
//        var dayAttendedLastEvent = 0
//        var answer = 0
//        for event in sortedEvents {
//            if dayAttendedLastEvent < event.first! {
//                print("Upper for loop \(event)")
//                dayAttendedLastEvent = event.first!
//                print(dayAttendedLastEvent)
//            }
//            if dayAttendedLastEvent <= event.last! {
//                print(event)
//                dayAttendedLastEvent += 1
//                answer += 1
//            }
//        }
//        return answer
//         This approach would fail for [[2, 3], [2, 3], [1, 5], [1, 5], [1, 5]] as
//         day 2: [2,3]
//         day 3: [2,3]
//         day 4: [1,5]
//         day 5: [1,5]
//         ----> Could have attend this event on day1: [1,5] but our code won't be able to detect that.
//         Ah, the classic "can we do it without a priority queue" dilemma. You're absolutely right to question this, especially since this is an interview. Let’s critically analyze whether we can resolve this **without a priority queue** while still maximizing the number of events attended.
//
//         The key challenge here is **dynamically selecting the earliest available day** for each event, especially when events overlap significantly. A **priority queue** naturally handles this by efficiently managing the smallest available end day. Without it, you'd need to simulate available days in some other way.
//
//         ### Possible Alternatives Without a Priority Queue:
//         1. **Boolean Array for Day Tracking**:
//            - You could use a boolean array (or set) to track which days have already been used, iterating from the event's start day to its end day to find the first available day.
//            - This avoids the need for a priority queue but comes at the cost of increased time complexity due to the repeated scans for each event.
//
//         2. **Sorting and Greedy Iteration**:
//            - While your current approach sorts the events by end day and tries to track the last attended day, it fails when earlier days are left unused (as in your example).
//            - If we modify this slightly to explicitly check all days between `startDay` and `endDay` for each event, we might resolve the issue—but this essentially simulates a priority queue inefficiently.
//
//         ---
//
//         ### Trade-Offs:
//         - Without a priority queue, you'll likely sacrifice efficiency for simplicity. The time complexity could degrade to **O(n × max(endDay - startDay))**, which is impractical for large inputs.
//         - With a priority queue, you maintain optimal time complexity of **O(n log n)**, which is crucial given the constraints.
//
//         ---
//
//         So, to directly answer your question:
//         **Can we resolve this without a priority queue?** Technically, yes. But practically, the priority queue is the optimal tool for this job. Without it, you're reinventing a less efficient wheel.
//         */
//        
//        var sortedEvents = events.sorted { $0.first! < $1.first! } // so that we process the events in an order of their happening
//        var minHeap = MinHeap() // so that we take that event which has earliest ending day
//        var currDay = sortedEvents[0].first!
//        var i = 0
//        var answer = 0
//        while i < sortedEvents.count || !minHeap.isEmpty {
//            if minHeap.isEmpty {
//                currDay = sortedEvents[i].first!
//            }
//            
//            // push all the events happening on this currDay
//            while ( i<sortedEvents.count && sortedEvents[i].first! == currDay){
//                minHeap.insert(sortedEvents[i].last!)
//                i += 1
//            }
//            
//            // process the event on currDay
//            if (!minHeap.isEmpty) {
//                answer += 1
//                _ = minHeap.extractMin()
//            }
//            
//            currDay += 1
//            while (!minHeap.isEmpty && minHeap.peek! < currDay){
//               _ = minHeap.extractMin()
//            }
//        }
//        return answer
//    }
//    
//    
//    func candy(_ ratings: [Int]) -> Int {
//        
//        /*
//         predecessor and successor approach , left to right and right to left
//         Your code is **correct!** 🎉
//
//         It implements the **two-pass greedy algorithm** effectively, ensuring that both the **left-to-right** and **right-to-left** conditions for candy distribution are satisfied. Here's why your solution works:
//
//         ### Key Points:
//         1. **Initialization:**
//            - Each child starts with 1 candy, satisfying the "at least one candy" rule.
//
//         2. **Left-to-Right Pass:**
//            - If a child has a higher rating than the previous child, they get one more candy than the previous child.
//
//         3. **Right-to-Left Pass:**
//            - If a child has a higher rating than the next child, their candy count is set to the maximum of the current value and `1 + candy[i+1]`. This ensures that both the left and right constraints are respected.
//
//         4. **Final Sum:**
//            - The total candies are calculated using a `reduce` function, which is efficient and clean.
//
//         ---
//
//         ### Example Walkthrough:
//         For `ratings = [1,0,2]`:
//         - **Left-to-right pass:** `[1, 1, 2]`
//         - **Right-to-left pass:** `[2, 1, 2]`
//         - **Total candies:** `2 + 1 + 2 = 5`
//
//         For `ratings = [1,2,2]`:
//         - **Left-to-right pass:** `[1, 2, 1]`
//         - **Right-to-left pass:** `[1, 2, 1]`
//         - **Total candies:** `1 + 2 + 1 = 4`
//
//         ---
//
//         ### Final Note:
//         Your inline comment is spot on—directly assigning `candyAssigned[i] = 1 + candyAssigned[i+1]` in the right-to-left pass would ignore the left-to-right condition, which is why using `max` is critical.
//
//         Great work! 🚀 If you have any follow-ups, feel free to ask.
//         */
//        
//        //print(ratings)
//             var candyAssigned = Array(repeating: 1, count: ratings.count)
//             for i in 1..<candyAssigned.count {
//                 if ratings[i] > ratings[i-1] {
//                     candyAssigned[i] = 1 + candyAssigned[i-1]
//                 }
//             }
//        //print(candyAssigned)
//             for i in stride(from: candyAssigned.count-2, through: 0, by: -1){
//                 if ratings[i] > ratings[i+1] {
//                     // candyAssigned[i] = 1 + candyAssigned[i+1] - This will be wrong since this is only considering the successor check condition, need to see predecessor check also
//                     candyAssigned[i] = max(candyAssigned[i], 1+candyAssigned[i+1])
//                 }
//              }
//        //print(candyAssigned)
//              return candyAssigned.reduce(0) { partialResult, val in
//         partialResult + val
//             }
//     }
//    
//    func numIslands(_ grid: [[Character]]) -> Int {
//        
//        var answer = 0
//        
//        let m = grid.count
//        let n = grid[0].count
//        
//        var visited = Array(repeating: Array(repeating: false, count: n), count: m)
//        
//        let isSafe: (Int, Int) -> Bool = { i, j in
//            (0..<m).contains(i) &&
//            (0..<n).contains(j) &&
//            grid[i][j] == "1" &&
//            !visited[i][j]
//        }
//        
//        var directions = [(0,1),(1,0),(0,-1),(-1,0)]
//        
//        var queue = [(Int,Int)]()
//        
//        let bfs: () -> Void =  {
//            while !queue.isEmpty {
//                var size = queue.count
//                while size != 0 {
//                    var (currRow, currCol) = queue.removeFirst()
//                    for direction in directions {
//                        var newRow = currRow + direction.0
//                        var newCol = currCol + direction.1
//                        if isSafe(newRow,newCol) {
//                            visited[newRow][newCol] = true
//                            queue.append((newRow,newCol))
//                        }
//                    }
//                    size -= 1
//                }
//            }
//        }
//        
//        for i in 0..<m {
//            for j in 0..<n {
//                if isSafe(i,j) {
//                    visited[i][j] = true
//                    queue.append((i,j))
//                    bfs()
//                    answer += 1
//                }
//            }
//        }
//        
//        return answer
//    }
//    
//}
//
//let solution = Solution()
//print(solution.candy([1,2,87,87,87,2,1]))
class MinHeap {
    var minHeap : [(Int,Int)] = []
    var count : Int {
        minHeap.count
    }
    var isEmpty : Bool {
        minHeap.isEmpty
    }
    var peek : (Int,Int)? {
        minHeap.first
    }

    func insert(_ x: (Int,Int)) {
        //Heapify up
        minHeap.append(x)
        var index = count - 1
        while index > 0 {
            var parentIdx = ( index - 1 )/2
            if minHeap[parentIdx].1 > minHeap[index].1 {
                minHeap.swapAt(parentIdx,index)
                index = parentIdx
            } else {
                break
            }
        }
    }

    func extractMin() -> (Int,Int)? {
        guard !isEmpty else {return nil}
        var answer = minHeap[0]
        if count > 1 {
            minHeap[0] = minHeap.removeLast()
            var parentIdx = 0
            //Heapify down
            while true {
                var leftChildIdx = 2 * parentIdx + 1
                var rightChildIdx = 2 * parentIdx + 2
                var smallestIdx = parentIdx
                if leftChildIdx < count && minHeap[smallestIdx].1 > minHeap[leftChildIdx].1 {
                    smallestIdx = leftChildIdx
                }
                if rightChildIdx < count && minHeap[smallestIdx].1 > minHeap[rightChildIdx].1 {
                    smallestIdx = rightChildIdx
                }

                if parentIdx == smallestIdx {
                    break
                }
                
                minHeap.swapAt(parentIdx,smallestIdx)
                parentIdx = smallestIdx
            }
        } else {
            minHeap.removeLast()
        }
        return answer
    }
}


class Solution {
    func topKFrequent(_ nums: [Int], _ k: Int) -> [Int] {
        var frequency = [Int:Int]()
        for num in nums {
            frequency[num, default:0] += 1
        }
        var output = [Int]()
        var minHeap = MinHeap()
        for (key,freqValue) in frequency {
            
            if minHeap.isEmpty {
                minHeap.insert((key,freqValue))
                continue
            }
            
            if minHeap.peek!.1 < freqValue {
                minHeap.insert((key,freqValue))
                if minHeap.count > k {
                    _ = minHeap.extractMin()
                }
            }
        }
        
        for i in 0..<k {
            if let topValExist = minHeap.peek {
                _ = minHeap.extractMin()
                output.append(topValExist.0)
            }
        }
        output = output.sorted()
        return output
    }
    
    
    func groupAnagrams(_ strs: [String]) -> [[String]] {
//        var hashMap : [String:[Int]] = [:]
//        
//        for (index,word) in strs.enumerated() {
//            var sortedWord = String(word.sorted())
//            hashMap[sortedWord, default: []].append(index)
//        }
//        //print(hashMap)
//        var output = [[String]]()
//        for indices in hashMap.values {
//            var currString = [String]()
//            for index in indices {
//                currString.append(strs[index])
//            }
//            output.append(currString)
//        }
//        return output
        
//        var hashMap : [String:[String]] = [:]
//        
//        for (index,word) in strs.enumerated() {
//            var sortedWord = String(word.sorted())
//            hashMap[sortedWord, default: []].append(word)
//        }
//        //print(hashMap)
//        var output = [[String]]()
//        for strings in hashMap.values {
//            var currString = [String]()
//            for str in strings {
//                currString.append(str)
//            }
//            output.append(currString)
//        }
//        return output
        
        // If the input consists of only small alphabet letters : a...z. character count array of size 26 and use index as ch.asciiValue - Character("a").asciiValue
        var hashMap : [[Int]:[String]] = [:]
        
        for word in strs {
            var characterCount = Array(repeating: 0, count: 26)
            for ch in word {
                characterCount[Int(ch.asciiValue! - Character("a").asciiValue!)] += 1
            }
            hashMap[characterCount,default: []].append(word)
        }
        var output = [[String]]()
        for strings in hashMap.values {
            var currString = [String]()
            for str in strings {
                currString.append(str)
            }
            output.append(currString)
        }
        return output

    }
}


"*Remember this is an interivew*"

let solution = Solution()
//print(solution.topKFrequent([1,1,1,2,2,3], 2))
solution.groupAnagrams(["eat","tea","tan","ate","nat","bat"])
