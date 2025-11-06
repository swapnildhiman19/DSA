//: [Previous](@previous)

import Foundation

//class Worker {
//    private var completionHandler: (()->Void)?
//    
//    func startLongJob(completion: @escaping ()->Void){
//        print("Worker has recieved the completion handler, starting it's job")
//        self.completionHandler = completion
//        
//        DispatchQueue.global().asyncAfter(deadline:.now()+2) {
//            print("Worker job has been completed")
//            self.completionHandler?()
//        }
//    }
//    
//    deinit {
//        print("Worker has been dinitialized")
//    }
//}
//
//class MyViewController {
//    private let worker = Worker()
//    
//    func startWork(){
//        worker.startLongJob {
//            [weak self] in
//            guard let self = self else {
//                print("MyViewController is no longer in the memory, unable to update the UI")
//                return
//            }
//            self.updateUI()
//        }
//    }
//    
//    func updateUI(){
//        print("Updating the UI of MyViewController")
//    }
//    
//    deinit{
//        print("MyViewController has been deinitialized")
//    }
//}

//var myViewController : MyViewController? = MyViewController()
//myViewController?.startWork()

//DispatchQueue.global().asyncAfter(deadline: .now()+1) {
//    print("MyViewController is no longer in the memory")
//    myViewController = nil
//}

//class ProgressTracker {
//    func startDownload(onProgress: @escaping ((Int)->Void)) {
//        print("Starting download...")
//
//        // Call the completion multiple times as progress updates
//        for progress in 1...5 {
//            DispatchQueue.global().asyncAfter(deadline: .now() + Double(progress)) { [weak self] in
//                print("Progress: \(progress * 20)%")
//                onProgress(progress*20)  // Call it repeatedly
//            }
//        }
//    }
//    
//    deinit {
//        print("ProgressTracker: deinitialized")
//    }
//}
//
//class DownloadViewController {
//    let tracker = ProgressTracker()
//    
//    func startDownload(){
//        tracker.startDownload { [weak self] progress in
//            guard let self = self else { return }
//            print("DownloadViewController: Updating progress bar to \(progress)%")
//        }
//    }
//    
//    deinit {
//        print("DownloadViewController: deinitialized")
//    }
//}
//
//// Usage
//var vc: DownloadViewController? = DownloadViewController()
//vc?.startDownload()
//

//class ProgressTracker {
//    var onProgress: ((Int) -> Void)?
//    
//    // Set the callback first
//    func setProgressHandler(_ handler: @escaping (Int) -> Void) {
//        self.onProgress = handler
//    }
//    
//    // Start later
//    func startDownload() {
//        for progress in 1...5 {
//            DispatchQueue.global().asyncAfter(deadline: .now() + Double(progress)) {
//                self.onProgress?(progress * 20)
//            }
//        }
//    }
//}
//
//let tracker = ProgressTracker()
//tracker.setProgressHandler { progress in
//    print("Progress right now : \(progress)%")
//}
//tracker.startDownload()

import Foundation

//class Worker {
//    // Store it ONLY if you need to cancel
//    private var completionHandler: (() -> Void)?
//    private var isCancelled = false
//    
//    func startLongJob(completion: @escaping () -> Void) {
//        print("Worker: received closure, storing to allow cancellation")
//        self.completionHandler = completion
//        self.isCancelled = false
//        
//        DispatchQueue.global().asyncAfter(deadline: .now() + 2) { [weak self] in
//            guard let self = self else { return }
//            
//            // Check if job was cancelled
//            if self.isCancelled {
//                print("Worker: job was cancelled, NOT calling completion")
//                self.completionHandler = nil
//                return
//            }
//            
//            print("Worker: job done, calling completion")
//            self.completionHandler?()
//            self.completionHandler = nil  // Clean up after use
//        }
//    }
//    
//    // This is WHY we store it - to cancel from another method!
//    func cancelJob() {
//        print("Worker: cancelling job")
//        isCancelled = true
//        completionHandler = nil  // Clear to prevent callback
//    }
//    
//    deinit {
//        print("Worker: deinitialized")
//    }
//}
//
//class MyViewController {
//    let worker = Worker()
//    
//    func startWork() {
//        print("MyViewController: Starting work")
//        worker.startLongJob { [weak self] in
//            guard let self = self else {
//                print("MyViewController: I'm already gone")
//                return
//            }
//            self.updateUI()
//        }
//    }
//    
//    func userTappedCancel() {
//        print("MyViewController: User cancelled")
//        // Can only cancel because Worker stored the closure!
//        worker.cancelJob()
//    }
//    
//    func updateUI() {
//        print("MyViewController: UI Updated!")
//    }
//    
//    deinit {
//        print("MyViewController: deinitialized")
//    }
//}
//
//// DEMO 1: Normal completion
//print("=== DEMO 1: Normal Completion ===")
//var vc1: MyViewController? = MyViewController()
//vc1?.startWork()
//
//DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
//    vc1 = nil
//    
//    // DEMO 2: Cancelled job
//    print("\n=== DEMO 2: Cancelled Job ===")
//    var vc2: MyViewController? = MyViewController()
//    vc2?.startWork()
//    
//    // User cancels after 1 second
//    DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
//        vc2?.userTappedCancel()
//    }
//    
//    DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
//        vc2 = nil
//    }
//}
//
//import PlaygroundSupport
//PlaygroundPage.current.needsIndefiniteExecution = true


//class Worker {
//    // Store it ONLY if you need to cancel
//    // private var completionHandler: (() -> Void)?
//    private var isCancelled = false
//    
//    func startLongJob(completion: @escaping () -> Void) {
//        print("Worker: received closure, storing to allow cancellation")
//        // self.completionHandler = completion
//        self.isCancelled = false
//        
//        DispatchQueue.global().asyncAfter(deadline: .now() + 2) { [weak self] in
//            guard let self = self else { return }
//            
//            // Check if job was cancelled
//            if self.isCancelled {
//                print("Worker: job was cancelled, NOT calling completion")
//                //self.completionHandler = nil
//                return
//            }
//            
//            print("Worker: job done, calling completion")
//            //self.completionHandler?()
//            completion()
//            //self.completionHandler = nil  // Clean up after use
//        }
//    }
//    
//    // This is WHY we store it - to cancel from another method!
//    func cancelJob() {
//        print("Worker: cancelling job")
//        isCancelled = true
//        // completionHandler = nil  // Clear to prevent callback
//    }
//    
//    deinit {
//        print("Worker: deinitialized")
//    }
//}
//
//class MyViewController {
//    let worker = Worker()
//
//    func startWork() {
//        print("MyViewController: Starting work")
//        worker.startLongJob { [weak self] in
//            guard let self = self else {
//                print("MyViewController: I'm already gone")
//                return
//            }
//            self.updateUI()
//        }
//    }
//
//    func userTappedCancel() {
//        print("MyViewController: User cancelled")
//        // Can only cancel because Worker stored the closure!
//        worker.cancelJob()
//    }
//
//    func updateUI() {
//        print("MyViewController: UI Updated!")
//    }
//
//    deinit {
//        print("MyViewController: deinitialized")
//    }
//}
//
//// DEMO 1: Normal completion
//print("=== DEMO 1: Normal Completion ===")
//var vc1: MyViewController? = MyViewController()
//vc1?.startWork()
//
//DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
//    vc1 = nil
//
//    // DEMO 2: Cancelled job
//    print("\n=== DEMO 2: Cancelled Job ===")
//    var vc2: MyViewController? = MyViewController()
//    vc2?.startWork()
//
//    // User cancels after 1 second
//    DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
//        vc2?.userTappedCancel()
//    }
//
//    DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
//        vc2 = nil
//    }
//}
//
//import PlaygroundSupport
//PlaygroundPage.current.needsIndefiniteExecution = true

class Worker {
    //private var completionHandler: (() -> Void)?  // Need to store!
    
    func startLongJob(completion: @escaping () -> Void) {
        //self.completionHandler = completion
        
        // Multiple async operations
        checkDatabase(completion)
        checkNetwork(completion)
        checkFiles(completion)
    }
    
    // Each of these methods needs access to the same completion
    private func checkDatabase(_ completion: @escaping () -> Void) {
        DispatchQueue.global().asyncAfter(deadline: .now() + 1) { [weak self] in
            print("Database checked")
            self?.tryComplete(completion)  // Calls stored completion
        }
    }
    
    private func checkNetwork(_ completion: @escaping () -> Void) {
        DispatchQueue.global().asyncAfter(deadline: .now() + 2) { [weak self] in
            print("Network checked")
            self?.tryComplete(completion)  // Calls stored completion
        }
    }
    
    private func checkFiles(_ completion: @escaping () -> Void) {
        DispatchQueue.global().asyncAfter(deadline: .now() + 3) { [weak self] in
            print("Files checked")
            self?.tryComplete(completion)  // Calls stored completion
        }
    }
    
    private var completedTasks = 0
    
    private func tryComplete(_ completion: @escaping () -> Void) {
        completedTasks += 1
        if completedTasks == 3 {
            completion()  // ✅ MUST be stored for this access!
            //completionHandler = nil
        }
    }
}
