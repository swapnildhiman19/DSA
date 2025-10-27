import Foundation

class Worker {
    var completionHandler : (() -> Void)?
    
    func startLongJob(completion: @escaping () -> Void ) {
        print("Worker received the closure, storing for later")
        self.completionHandler = completion
        
        DispatchQueue.global().asyncAfter(deadline: .now() + 2.0) {
            [weak self] in
            print("Worker job done, time to call back")
            self?.completionHandler?()
        }
    }
    
    deinit {
        print("Worker: deinitalized")
    }
}

class MyViewController {
    let worker = Worker()
    
    // Creating a function to start work
    func startWork() {
        worker.startLongJob {
            //Need to use [weak self]
            [weak self] in
            
            guard let self else {
                print("MyViewController is already deallocated")
                return
            }
            
            self.updateUI()
        }
    }
    
    func updateUI() {
        print("MyViewController: UI updated")
    }
    
    deinit {
        print("MyViewController: deinitalized")
    }
}

var vc : MyViewController? = MyViewController()
vc?.startWork()

DispatchQueue.global().asyncAfter(deadline: .now() + 1.0, execute: {
    print("Main: Setting vc to nil")
    vc = nil
} )
