//import Foundation
//////MARK: Strong Reference Understanding
/////*
//// 1 Strong Reference
//// 
////class Bird {
////    var nest: Nest? // Strong reference
////    deinit { print("Bird deallocated") }
////}
////
////class Nest {
////    var bird: Bird? // Strong reference
////    deinit { print("Nest deallocated") }
////}
////
////// ARCbird = 0, ARCnest = 0
////
////var b: Bird? = Bird()         // ARCbird = 1 (b points to Bird instance)
////var n: Nest? = Nest()         // ARCnest = 1 (n points to Nest instance)
////
////b!.nest = n                   // ARCnest = 2 (n, b!.nest both point to Nest)
////n!.bird = b                   // ARCbird = 2 (b, n!.bird both point to Bird)
////
////b = nil                       // ARCbird = 1 (still held by n!.bird), ARCnest = 2
////n = nil                       // ARCnest = 1 (still held by b!.nest), ARCbird = 1
////
////// Both ARCbird and ARCnest stay at 1: no deinit is called, memory is leaked! -> Loop is getting created
////
////
////class Bird {
////    var nest: Nest? // strong reference
////    deinit { print("Bird deallocated") }
////}
////
////class Nest {
////    var bird: Bird? // strong reference
////    deinit { print("Nest deallocated") }
////}
////
////// ARCbird = 0, ARCnest = 0
////
////var b: Bird? = Bird()          // ARCbird = 1
////var n: Nest? = Nest()          // ARCnest = 1
////
////b!.nest = n                    // ARCnest = 2 (b!.nest holds n), ARCbird = 1
////n!.bird = b                    // ARCbird = 2 (n!.bird holds b), ARCnest = 2
////
////b = nil                        // ARCbird = 1 (only n!.bird holds b), ARCnest = 2
////n!.bird = b                    // n!.bird = nil; ARCbird = 0 (no refs to b anymore), ARCnest = 2
////
////n = nil                        // ARCnest = 1 (only b!.nest holds n, but b is now nil), so now ARCnest = 0
////
////// Now both ARCbird and ARCnest are 0, both are deallocated; you see:
////// Bird deallocated
////// Nest deallocated
////
////By setting n!.bird = b (when b is already nil), you set nest's bird to nil, removing the last strong reference to Bird, so Bird is deallocated immediately. The only thing holding Nest was b!.nest, but since b is nil, nothing points to Nest now—so Nest deallocates.
////
////class Bird {
////    var nest: Nest? // strong reference
////    deinit { print("Bird deallocated") }
////}
////
////class Nest {
////    var bird: Bird? // strong reference
////    deinit { print("Nest deallocated") }
////}
////
////// ARCbird = 0, ARCnest = 0
////
////var b: Bird? = Bird()          // ARCbird = 1 (b -> Bird1)
////var n: Nest? = Nest()          // ARCnest = 1
////
////b!.nest = n                    // ARCnest = 2 (b!.nest -> n), ARCbird = 1
////n!.bird = b                    // ARCbird = 2 (n!.bird -> Bird1), ARCnest = 2
////
////b = nil                        // ARCbird = 1 (still n!.bird -> Bird1), ARCnest = 2
////
////n!.bird = Bird()               //
////// (1) n!.bird's old value (Bird1) ARCbird = 0 (deinit)
//////     print("Bird deallocated")
////// (2) n!.bird now points to Bird2. ARCbird(Bird2) = 1 (from n!.bird)
////
////n = nil                        // ARCnest = 1 (b!.nest -> n, but b = nil so gone, ARCnest = 0, deinit)
////// n!.bird releases Bird2, ARCbird(Bird2) = 0, Bird2 deinit
////
////// You see:
////// Bird deallocated   <-- (from releasing Bird1)
////// Nest deallocated
////// Bird deallocated   <-- (from releasing Bird2)
////
//// What's happening?
//// n!.bird = Bird() breaks the strong reference to the original Bird object (held by Nest), so Bird1 deallocates first.
////
//// The new temporary Bird object (Bird2) is only referenced by n!.bird.
////
//// After n = nil, Nest deallocates, and now Bird2 also deallocates because nothing holds it anymore.
////
//// Key ARC Rule Shown:
//// Assigning a new object to a strong reference property first releases the old object (possibly triggering deinit if its count drops to zero) and then increments the count of the new object.
////
//// Setting it to nil (n!.bird = b, b == nil) releases the reference from n!.bird, so Bird reference count drops to zero → deinit.
////*/
////
////
//////MARK: Weak Reference
/////*
////class Bird {
////    var nest: Nest? // Strong reference
////    deinit { print("Bird deallocated") }
////}
////
////class Nest {
////    weak var bird: Bird? // Weak reference (does not increase ARCbird)
////    deinit { print("Nest deallocated") }
////}
////
////// ARCbird = 0, ARCnest = 0
////
////var b: Bird? = Bird()         // ARCbird = 1 (b points to Bird instance)
////var n: Nest? = Nest()         // ARCnest = 1 (n points to Nest instance)
////
////b!.nest = n                   // ARCnest = 2 (n, b!.nest both point to Nest)
////n!.bird = b                   // ARCbird = 1 (still just 'b') (bird is weak, does NOT increase ARC)
////
////b = nil                       // ARCbird = 0 (no strong refs, Bird deallocated, n!.bird is now nil)
////                              // ARCnest = 2 (n, b!.nest since Bird out of memory, b!.nest is gone)
////n = nil                       // ARCnest = 1->0 (now no strong refs, Nest deallocated)
////
////// Both classes' deinit print: NO MEMORY LEAK!
////
////*/
////
////
/////*
//// Closures
//// */
////
////// 1 Capturing Values
////var len = 4
////let myClosure : (Int) -> Int = { x in
////    x+len
////}
////
////debugPrint(myClosure(10)) // 14
////
////len = 10
////
////debugPrint(myClosure(10)) // 20 -> It captures the fresh value of var len
////
////// Non Escaping closure
////
////func doSomethingImmediately(_ action: ()->Void) {
////    print("Start")
//////    DispatchQueue.main.asyncAfter(deadline: .now()+1.0) {
//////        action()
//////    } This is not possible because execute closure expects to have escaping closure in it
//////    action()
////    print("End")
////}
////
////doSomethingImmediately {
////    print("Action closure running")
////}
////
//////Output:
//////Start
//////Action closure running
//////End
////
////print(" ")
////
////
////// Escaping closure
////func doSomethingLater(_ action: @escaping ()->Void){
////    print("Start")
////    DispatchQueue.main.asyncAfterUnsafe(deadline: .now()+1.0) {
////        action()
////    }
//////    action()
////    print("End")
////}
////
////doSomethingLater {
////    print("Action Closure Running")
////}
////
////// Output:
//////Start
//////End
////// < one second passes >
//////Action Closure Running
////
////// Synchronous completion handler
////
////func add(_ a: Int, _ b: Int, _ completion: (Int)->()) {
////    let result = a+b
////    completion(result)
////}
////
////add(5, 6) { sum in
////    print("The sum here is \(sum)")
////}
////
////// Async custom completion handler
////
////func fetchUsername(completion: @escaping (String)->()){
////    DispatchQueue.main.asyncAfter(deadline: .now()+3.0) {
////        completion("Swapnil")
////    }
////}
////
////fetchUsername { name in
////    print("Name of the user is \(name)")
////}
////
//
////class Downloader {
////    
////    var completion: (()->Void)?
////    
////    func download(completion: @escaping ()->Void){
//////        self.completion = completion
////        DispatchQueue.main.asyncAfterUnsafe(deadline: .now()+2.0) {
////            completion()
////        }
////    }
////    
////    deinit{
////        print("downloader deinitialized")
////    }
////    
////}
////
////class ViewController {
////    
////    let downloader = Downloader()
////    
////    func fetchData(){
////        downloader.download {
////            self.updateUI()
////        }
////    }
////    
////    func updateUI(){
////        print("UI updated")
////    }
////    
////    deinit {
////        print("viewcontroller deinitialized")
////    }
////    
////}
////
////var vc : ViewController? = ViewController()
////vc!.fetchData()
////vc = nil
//
////import Foundation
////
////@MainActor
////class Worker {
////    // This property stores the closure for later (can create a retain cycle!)
////    var completion: (() -> Void)?
////
////    // This method accepts a closure and stores it, instead of using it right away
////    func startLongJob(completion: @escaping () -> Void) {
////        print("Worker: received closure, storing for later...")
////        self.completion = completion
////
////        // Simulate an async job (e.g. network, file download)
////        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
////            print("Worker: job done, time to call back!")
////            // Actually call the closure from the property
////            self?.completion?()
////        }
////    }
////
////    deinit {
////        print("Worker: deinitialized")
////    }
////}
////
////@MainActor
////class MyViewController {
////    let worker = Worker()
////
////    func startWork() {
////        // Pass a closure to the worker, which uses 'self' inside
////        worker.startLongJob { [weak self] in
////            // Using [weak self] prevents the closure from keeping MyViewController alive
////            guard let self = self else {
////                print("MyViewController: I'm already gone, not updating UI")
////                return
////            }
////            self.updateUI()
////        }
////    }
////
////    func updateUI() {
////        print("MyViewController: UI Updated!")
////    }
////
////    deinit {
////        print("MyViewController: deinitialized")
////    }
////}
////
////// MAIN — simulate the controller coming and going
////var vc: MyViewController? = MyViewController()
////vc?.startWork()
////
////// Simulate the controller being removed before the job finishes:
////DispatchQueue.global().asyncAfter(deadline: .now() + 4) {
////    print("Main: Setting vc to nil (remove ViewController)")
////    DispatchQueue.main.async {
////        vc = nil
////    }
////}
////
////// Keep the playground running long enough to see print output
////import PlaygroundSupport
////PlaygroundPage.current.needsIndefiniteExecution = true
//
////func makeIncrement(amount:Int) -> (()->Int){
////    var total = 0
////    func incrementer () -> Int {
////        print("Value of total right now is \(total)")
////        total = total + amount
////        return total
////    }
////    return incrementer
////}
////
////let incrementByTwo = makeIncrement(amount: 2)
////print(incrementByTwo())
////print(incrementByTwo())
//
///*
// Excellent question! This confusion is **very common** when learning about closures **and their ability to capture variables** in Swift.
//
// Let's break it down—**first in simple layman's language**, then with a detailed explanation of exactly how Swift "remembers" the `total` variable.
//
// ***
//
// ## Layman's Explanation
//
// Imagine you call `makeIncrement(amount: 2)` and get back a function ("incrementer") that, whenever you call it, adds 2 to a "hidden" counter `total` and returns the result.
//
// **But where does `total` live?**
// It's not a global; it's not passed into the function; it's not a static; yet the function "remembers" it every time you call it.
//
// **Closure Magic:**
// When you return a function from inside another function, Swift **"captures"** the variables from that parent function (like `total`) and keeps them *alive* inside the closure, even after the original function (`makeIncrement`) has returned!
//
// **So:**
// - Each time you call `incrementByTwo()`, it’s using its own private copy of `total`, *remembered* just for it.
//
// ***
//
// ## Step-by-Step Sequence
//
// Here’s what happens in your code:
//
// 1. **You call:**
//    ```swift
//    let incrementByTwo = makeIncrement(amount: 2)
//    ```
//    - `makeIncrement` runs, sets `total = 0`.
//    - Returns the inner function `incrementer`, *which "captures"* `total` and `amount`.
//
// 2. **Now you call:**
//    ```swift
//    print(incrementByTwo()) // 2
//    ```
//    - The closure prints: `"Value of total right now is 0"`
//    - It adds `2` to `total`, now `total` is `2`
//    - Returns `2`
//
// 3. **Call it again:**
//    ```swift
//    print(incrementByTwo()) // 4
//    ```
//    - The closure prints: `"Value of total right now is 2"`
//    - Adds `2` to `total`, now `total` is `4`
//    - Returns `4`
//
// **So, Swift "remembers"** total's value inside the returned closure function.
//
// ***
//
// ## Visualization
//
// Let’s visualize the relationship:
//
// ```
// incrementByTwo ----> closure code (has its OWN total variable)
//                           |
//                           --> total = 0 (captured at creation)
//                           --> amount = 2 (captured at creation)
// ```
// Every time you call `incrementByTwo()`, it uses and updates its own private `total`.
//
// ***
//
// ## Proof: Each Closure Has Its Own Captured Variables
//
// Try this!
// ```swift
// let incByTwo = makeIncrement(amount: 2)
// let incByFive = makeIncrement(amount: 5)
// print(incByTwo())  // prints 2
// print(incByTwo())  // prints 4
// print(incByFive()) // prints 5 -- separate total!
// print(incByFive()) // prints 10
// ```
// - Each closure remembers its own captured `total` and `amount` independently.
//
// ***
//
// ## Swift Closure Capture in Detail
//
// - When you define a function inside another function (a nested function or closure),
// - And return that inner function,
// - Swift stores not just the code, but also *all variables referenced from the outer function* at the time.
// - These variables are stored with the function as long as you keep a reference to the closure.
//
// **This is known as a closure ‘capturing’ its enclosing environment.**
//
// ***
//
// ## TL;DR
//
// - When you call `makeIncrement`, you get a function ("closure") that "remembers" its own private variables (`total`, `amount`) even when the original function has finished.
// - Each returned closure instance gets **its own private captured memory** for those variables.
//
// ***
//
// If you want to "see" this, use the visualization above, or ask for a playground demonstration—you'll see every closure keeps its own little “backpack” of captured variables! Let me know if you want further code experiments to cement this concept!
// */
//
//var adder : (Int,Int) -> Int = {
//    (a,b) in
//    return a+b
//}
//
//print(adder(2,3))
//
//func makeCounter() -> () -> Int {
//    var count = 0
//    return {
//        count += 1
//        return count
//    }
//}
//
//let counter = makeCounter()
//print(counter())
//print(counter())
////It will print 1 and 2 , because it captures value of var present in parent function here it is count
//
////Non escaping closures which are also our default closures in swift is basically gets executed before the function in which this was passed as argument returns, so everything is happening synchronously here
//// Escaping closures : @escaping , need [weak self] to not retain cycle and have memory leak, we use this if we want the to store the data , like URL networking session where the parent function can return but this will run and can be executed after that.
//
//// Yes since it's a strong reference there will be memory leak
//
//let array = [1,2,3,20]
//var newArray = array.filter {
//    $0 > 10
//}
//print(newArray)
//
//
////func performTwice(action: (String) -> Void)
////means it takes a closure named action as input parameter which will take string as input and returns nothing
//
////func makeAdder(value: Int) -> (Int) -> Int {
////    var sum = 0
////    return { [sum, value] in
////        sum + value
////    }
////}
////
////let adder = makeAdder(value: 7)
////print(adder(3))
////print(adder(10))
//
//// It will print 3, 10 ( which is 3 + 7 ) - I don't know the answer probably might be wrong
//
//func downloadData(completion: @escaping ()->Void){
//    DispatchQueue.main.asyncAfter(deadline: .now()+2) {
//        completion()
//    }
//}
//
//// [weak self] we use in swift to not increase the ARC counter [unowned self] also won't increase the counter but we need to sure that var to whom it is pointing to should always stay in memory when trying to access otherwise it might crash

import Foundation

//func makeAdderValue(value: Int) -> (Int) -> Int {
//    var sum = 0
//    return { [sum, value] (x: Int) -> Int in
//        // 'sum' and 'value' used in here are COPIED AT CREATION
//        return sum + value
//    }
//}
//
//let adder = makeAdderValue(value: 7)
//print(adder(3))   // always prints 0 + 7 = 7
//print(adder(10))  // always prints 0 + 7 = 7

func capture() -> [()->Void] {
    var output = [()->Void]()
    for i in 0...2 {
        var value = i
//        func valueCaptured () -> Int {
//            print("Closure \(i): My captured value is \(i)")
//            return value
//        }
        output.append { [index = i] in
            print("Closure \(index): My captured value is \(index)")
        }
    }
    return output
}

let closures = capture()
for closure in closures {
    closure()
}
