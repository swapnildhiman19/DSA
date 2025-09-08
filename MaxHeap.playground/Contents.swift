import Foundation

// MaxHeap is a complete BT except the last level
class MaxHeap {
    var heap: [Int] = []
    
    var isEmpty: Bool {
        heap.isEmpty
    }
    
    var count : Int {
        heap.count
    }
    
    var peek: Int? {
        heap.first
    }
    
    // Heapify up
    func insert(_ value :Int) {
        heap.append(value)
        var idx = heap.count - 1
        while idx > 0 {
            let parentIdx = (idx - 1)/2
            if heap[parentIdx] < heap[idx] {
                heap.swapAt(parentIdx, idx)
                idx = parentIdx
            } else {
                break
            }
        }
    }
    
    // Remove the max Element : Heapify Down
    func extractMax() -> Int? {
        guard  !heap.isEmpty else {
            return nil
        }
        
        if heap.count == 1 {
           return heap.removeFirst()
        }
        
        let maxVal = heap[0]
        heap[0] = heap.removeLast()
        var idx = 0
        
        while true {
            var leftChildIdx = (2 * idx) + 1
            var rightChildIdx = (2 * idx) + 2
            
            var largest = idx
            
            if leftChildIdx < heap.count && heap[leftChildIdx] > heap[largest] {
                largest = leftChildIdx
            }
            
            if rightChildIdx < heap.count && heap[rightChildIdx] > heap[largest] {
                largest = rightChildIdx
            }
            
            if largest == idx {
                // at correct place we have reached
                break
            }
            
            heap.swapAt(idx, largest)
            idx = largest
        }
        return maxVal
    }
}

var heap = MaxHeap()
heap.insert(8)
heap.insert(10)
heap.insert(4)
heap.insert(15)
heap.insert(7)
print(heap.heap)

print(heap.extractMax()!)
print(heap.heap)
