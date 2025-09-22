import Foundation

//class ArrayStack<T> {
//
//    private var stack: [T]
//
//    init() {
//        stack = []
//    }
//
//    func push(_ x: T) {
//        stack.append(x)
//    }
//
//    func pop() -> T? {
//        guard !isEmpty() else {
//            return nil
//        }
//        return stack.removeLast()
//    }
//
//    func top() -> T? {
//        guard !isEmpty() else {
//            return nil
//        }
//        return stack.last
//    }
//
//    func isEmpty() -> Bool {
//        stack.isEmpty
//    }
//}

//class ArrayQueue<T> {
//
//    private var queue: [T]
//
//    init() {
//        queue = []
//    }
//
//    func push(_ x: T) {
//        queue.append(x)
//    }
//
//    func pop() -> T? {
//        guard !isEmpty() else {
//            return nil
//        }
//        return queue.removeFirst()
//    }
//
//    func top() -> T? {
//        guard !isEmpty() else {
//            return nil
//        }
//        return queue.first
//    }
//
//    func isEmpty() -> Bool {
//        queue.isEmpty
//    }
//}

/*
 Key Points:
 O(1) enqueue and dequeue: No shifting of array elements on every pop.

 Memory cleanup: The array is trimmed occasionally to save space.

 Storage is reused: Only physical removal when unused slots become significant.
 
 Using head counter and to return for majority of times, if storage becomes too long then we delete that prefix of head elements and reset the array
 */
//class ArrayQueue<T> {
//    private var queue: [T] = []
//    private var head: Int = 0
//
//    // Enqueue (push)
//    func push(_ x: T) {
//        queue.append(x)
//    }
//
//    // Dequeue (pop)
//    func pop() -> T? {
//        guard head < queue.count else { return nil }
//        let element = queue[head]
//        head += 1
//
//        // Periodically clean up storage if head becomes large
//        if head > 50 && head * 2 > queue.count {
//            queue.removeFirst(head) //Remove first head elements from the array
//            head = 0
//        }
//
//        return element
//    }
//
//    // Peek (front/top)
//    func top() -> T? {
//        guard head < queue.count else { return nil }
//        return queue[head]
//    }
//
//    // Is Empty
//    func isEmpty() -> Bool {
//        head >= queue.count
//    }
//}


//
//// Stack implementation using Queue
//class MyStack {
//    var queue = [Int]()
//
//    init() {}
//    
//    // O(n) : operation
//    func push(_ x: Int) {
//        queue.append(x)
//        // Rotate all previous elements to the back
//        for _ in 0..<queue.count-1 {
//            queue.append(queue.removeFirst())
//        }
//    }
//    
//    // O(1) : operation
//    func pop() -> Int {
//        return queue.isEmpty ? -1 : queue.removeFirst()
//    }
//    
//    // O(1) : operation
//    func top() -> Int {
//        return queue.isEmpty ? -1 : queue.first!
//    }
//
//    func empty() -> Bool {
//        return queue.isEmpty
//    }
//}

//Queue using stack
class MyQueue {
    
    var stack = [Int]()

    init() {
        
    }
    
    func push(_ x: Int) {
        stack.append(x)
        for index in 0..<stack.count - 1 {
            stack.append(stack.remove(at: index))
        }
    }
    
    func pop() -> Int {
        return stack.isEmpty ? -1 : stack.removeLast()
    }
    
    func peek() -> Int {
        return stack.isEmpty ? -1 : stack.last!
    }
    
    func empty() -> Bool {
        return stack.isEmpty
    }
}
