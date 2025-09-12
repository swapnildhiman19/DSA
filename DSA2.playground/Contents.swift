//import Foundation
//
//struct IntPair : Hashable {
//    let x : Int
//    let y : Int
//}
//
//class Solution {
//    func maxSumCombinations(_ nums1: [Int], _ nums2: [Int], _ k : Int) -> [Int]{
//
//        var visited :  Set<IntPair> = []
//        var maxHeap = MaxHeap()
//        
//        let sortedNums1 = nums1.sorted(by: >)
//        let sortedNums2 = nums2.sorted(by: >)
//        
//        let n1 = sortedNums1.count
//        let n2 = sortedNums2.count
//        
//        var output = [Int]()
//        
//        maxHeap.insert([sortedNums1[0]+sortedNums2[0],0,0])  //contains sum, i, j
//        visited.insert(IntPair(x: 0, y: 0))
//        
//        for operation in 0..<k {
//            guard let maxElementInfo = maxHeap.extractMax() else {
//                break
//            }
//            var currBestSum = maxElementInfo[0]
//            var currI = maxElementInfo[1]
//            var currJ = maxElementInfo[2]
//            
//            // Next Possible combinations is i,j+1 or i+1,j
//            print("currBestSum \(currBestSum) currI \(currI) currJ \(currJ)")
//            
//            output.append(currBestSum)
//            
//            if currI + 1 < n1 {
//                if !visited.contains(IntPair(x: currI + 1, y: currJ)){
//                    visited.insert(IntPair(x: currI + 1, y: currJ))
//                    maxHeap.insert([sortedNums1[currI+1]+sortedNums2[currJ],currI+1,currJ])
//                }
//            }
//            
//            if currJ + 1 < n2 {
//                if !visited.contains(IntPair(x: currI, y: currJ+1)){
//                    visited.insert(IntPair(x: currI, y: currJ+1))
//                    maxHeap.insert([sortedNums1[currI]+sortedNums2[currJ+1],currI,currJ+1])
//                }
//            }
//        }
//        
//        return output
//    }
//    
//    func mergeKSortedArrays(_ nums:[[Int]], _ k: Int) -> [Int] {
//        var output = [Int]()
//        var minHeap = MinHeap()
//        
//        for i in 0..<k {
//            //insert the first element of K sorted arrays in minHeap
//            minHeap.insert((nums[i][0],i,0))
//        }
//        
//        while !minHeap.isEmpty {
//            guard  var (currMinValue, currMinRowIdx, currMinIdx) = minHeap.extractMin() else {
//                break
//            }
//            output.append(currMinValue)
//            if currMinIdx + 1 < nums[currMinRowIdx].count {
//                minHeap.insert((nums[currMinRowIdx][currMinIdx+1], currMinRowIdx, currMinIdx + 1))
//            }
//        }
//        
//        return output
//    }
//}
//
//class MinHeap {
//    var minHeap = [(Int,Int,Int)]() // This will contain the element and index like from which [[nums]] row this element exist from and in that which index I am on
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
//        minHeap.first?.0
//    }
//    
//    //Heapify up
//    func insert(_ x: (Int,Int,Int)) {
//        minHeap.append(x)
//        var idx = count - 1
//        
//        while idx > 0 {
//            var parentIdx = (idx - 1)/2
//            if minHeap[parentIdx].0 > minHeap[idx].0 {
//                minHeap.swapAt(parentIdx, idx)
//                idx = parentIdx
//            } else {
//                break
//            }
//        }
//    }
//    
//    //Heapify Down
//    func extractMin() -> (Int,Int,Int)? {
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
//                if leftChildIdx < count && minHeap[smallestIdx].0 > minHeap[leftChildIdx].0 {
//                    smallestIdx = leftChildIdx
//                }
//                
//                if rightChildIdx < count && minHeap[smallestIdx].0 > minHeap[rightChildIdx].0 {
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
//}
//
//class MaxHeap {
//    var maxHeap = [[Int]]()
//    
//    var isEmpty : Bool {
//        maxHeap.isEmpty
//    }
//    
//    var count: Int{
//        maxHeap.count
//    }
//    
//    var peek: Int? {
//        maxHeap.first?.first
//    }
//    
//    //Heapify up
//    func insert(_ x: [Int]) {
//        maxHeap.append(x)
//        var idx = maxHeap.count - 1
//        while idx > 0 {
//            var parentIdx = (idx - 1)/2
//            if maxHeap[parentIdx][0] < maxHeap[idx][0] {
//                maxHeap.swapAt(parentIdx, idx)
//                idx = parentIdx
//            } else {
//                break
//            }
//        }
//    }
//    
//    //Heapify Down
//    func extractMax() -> [Int]? {
//        guard !maxHeap.isEmpty else { return nil }
//        var answer = maxHeap[0]
//        
//        // If there's more than one element, move the last to the front and heapify down.
//        if maxHeap.count > 1 {
//            maxHeap[0] = maxHeap.removeLast()
//            var parentIdx = 0
//            
//            while true {
//                let leftChildIdx = 2 * parentIdx + 1
//                let rightChildIdx = 2 * parentIdx + 2
//                var largestIdx = parentIdx
//                
//                if leftChildIdx < maxHeap.count && maxHeap[leftChildIdx][0] > maxHeap[largestIdx][0] {
//                    largestIdx = leftChildIdx
//                }
//                
//                if rightChildIdx < maxHeap.count && maxHeap[rightChildIdx][0] > maxHeap[largestIdx][0] {
//                    largestIdx = rightChildIdx
//                }
//                
//                if largestIdx == parentIdx {
//                    break
//                }
//                
//                maxHeap.swapAt(parentIdx, largestIdx)
//                parentIdx = largestIdx
//            }
//        } else {
//            // If it's the last element, just remove it.
//            maxHeap.removeLast()
//        }
//        
//        return answer
//    }
//}
//
//let solution = Solution()
////print(solution.maxSumCombinations([10,2,1], [10,5,4,1], 6))
//print(solution.mergeKSortedArrays([[3,6,9], [2,4,6,8], [1,3,5,7]], 3))




class MedianFinder {
    var minHeap : MinHeap
    var maxHeap : MaxHeap

    init() {
        minHeap = MinHeap()
        maxHeap = MaxHeap()
    }
    
    func balance() {
        let minCount = minHeap.count
        let maxCount = maxHeap.count

        if maxCount - minCount > 1 {
            // Move the root of maxHeap to minHeap
            if let maxVal = maxHeap.extractMax() {
                minHeap.insert(maxVal)
            }
        } else if minCount - maxCount > 1 {
            // Move the root of minHeap to maxHeap
            if let minVal = minHeap.extractMin() {
                maxHeap.insert(minVal)
            }
        }
    }
    
    /*
    Insert Logic:

    Smaller element → maxHeap
    Larger element → minHeap
    Balance Logic:

    If one heap has more than one extra element compared to the other, transfer the top element from the larger heap to the smaller heap.
    Median Calculation Logic:

    If heaps are balanced → Median = Mean of tops of both heaps.
    If heaps are imbalanced → Median = Top of the larger heap.
    */


    func addNum(_ num: Int) {
    if maxHeap.isEmpty {
        maxHeap.insert(num)
        return
    }

    // Compare with the top of maxHeap
    if let maxSmallestElement = maxHeap.peek, num <= maxSmallestElement {
        maxHeap.insert(num)
    } else if let minLargestElement = minHeap.peek, num >= minLargestElement {
        // Compare with the top of minHeap
        minHeap.insert(num)
    } else {
        // Falls in between maxHeap and minHeap tops
        maxHeap.insert(num)
    }

    // Balance the heaps after insertion
    balance()
    }
    
    func findMedian() -> Double {
        // if heaps are balanced
        if minHeap.count == maxHeap.count {
            return Double(Double(minHeap.peek!+maxHeap.peek!)/2.0)
        }
        if minHeap.count > maxHeap.count {
            return Double(minHeap.peek!)
        }
        return Double(maxHeap.peek!)
    }
}


class MinHeap {
    var minHeap = [Int]()
    
    var isEmpty : Bool {
        minHeap.isEmpty
    }
    
    var count : Int {
        minHeap.count
    }
    
    var peek : Int? {
        minHeap.first
    }
    
    //Heapify up
    func insert(_ x: Int) {
        minHeap.append(x)
        var idx = count - 1
        
        while idx > 0 {
            var parentIdx = (idx - 1)/2
            if minHeap[parentIdx] > minHeap[idx] {
                minHeap.swapAt(parentIdx, idx)
                idx = parentIdx
            } else {
                break
            }
        }
    }
    
    //Heapify Down
    func extractMin() -> Int? {
        guard !isEmpty else {
            return nil
        }
        
        var answer = minHeap[0]
        
        if minHeap.count > 1 {
            minHeap[0] = minHeap.removeLast()
            var parentIdx = 0
            while true {
                var leftChildIdx = 2 * parentIdx + 1
                var rightChildIdx = 2 * parentIdx + 2
                
                var smallestIdx = parentIdx
                
                if leftChildIdx < count && minHeap[smallestIdx] > minHeap[leftChildIdx] {
                    smallestIdx = leftChildIdx
                }
                
                if rightChildIdx < count && minHeap[smallestIdx] > minHeap[rightChildIdx] {
                    smallestIdx = rightChildIdx
                }
                
                if parentIdx == smallestIdx {
                    break
                }
                
                minHeap.swapAt(smallestIdx, parentIdx)
                parentIdx = smallestIdx
            }
        } else {
            minHeap.removeLast()
        }
        return answer
    }
}


class MaxHeap {
    var maxHeap = [Int]()
    
    var isEmpty : Bool {
        maxHeap.isEmpty
    }
    
    var count: Int{
        maxHeap.count
    }
    
    var peek: Int? {
        maxHeap.first
    }
    
    //Heapify up
    func insert(_ x: Int) {
        maxHeap.append(x)
        var idx = maxHeap.count - 1
        while idx > 0 {
            var parentIdx = (idx - 1)/2
            if maxHeap[parentIdx] < maxHeap[idx] {
                maxHeap.swapAt(parentIdx, idx)
                idx = parentIdx
            } else {
                break
            }
        }
    }
    
    //Heapify Down
    func extractMax() -> Int? {
        guard !maxHeap.isEmpty else { return nil }
        var answer = maxHeap[0]
        
        // If there's more than one element, move the last to the front and heapify down.
        if maxHeap.count > 1 {
            maxHeap[0] = maxHeap.removeLast()
            var parentIdx = 0
            
            while true {
                let leftChildIdx = 2 * parentIdx + 1
                let rightChildIdx = 2 * parentIdx + 2
                var largestIdx = parentIdx
                
                if leftChildIdx < maxHeap.count && maxHeap[leftChildIdx] > maxHeap[largestIdx] {
                    largestIdx = leftChildIdx
                }
                
                if rightChildIdx < maxHeap.count && maxHeap[rightChildIdx] > maxHeap[largestIdx] {
                    largestIdx = rightChildIdx
                }
                
                if largestIdx == parentIdx {
                    break
                }
                
                maxHeap.swapAt(parentIdx, largestIdx)
                parentIdx = largestIdx
            }
        } else {
            // If it's the last element, just remove it.
            maxHeap.removeLast()
        }
        return answer
    }
}


/**
 * Your MedianFinder object will be instantiated and called as such:
 * let obj = MedianFinder()
 * obj.addNum(num)
 * let ret_2: Double = obj.findMedian()
 */

 /*
 Optimization using Binary Search if element [0,100]
 class MedianFinder {
    var frequency = [Int](repeating: 0, count: 101) // Frequency array for range [0, 100]
    var totalCount = 0                              // Total number of elements added

    init() {}

    func addNum(_ num: Int) {
        frequency[num] += 1  // Increment frequency for the number
        totalCount += 1      // Increment total count of numbers
    }

    func findMedian() -> Double {
        let targetLeft = (totalCount + 1) / 2       // Target index for the left median (for odd/even cases)
        var cumulativeCount = 0                    // Tracks the cumulative frequency

        var leftMedian = -1
        var rightMedian = -1

        for i in 0...100 {
            cumulativeCount += frequency[i]        // Update cumulative count

            if cumulativeCount >= targetLeft && leftMedian == -1 {
                leftMedian = i                     // First median (left side for even, or middle for odd)
            }
            if cumulativeCount >= (totalCount / 2 + 1) {
                rightMedian = i                    // Second median for even-sized streams
                break
            }
        }

        if totalCount % 2 == 0 {
            return Double(leftMedian + rightMedian) / 2.0 // Average for even-sized streams
        } else {
            return Double(leftMedian)                     // Middle for odd-sized streams
        }
    }
}
*/
