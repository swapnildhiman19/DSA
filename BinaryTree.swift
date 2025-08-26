//
//  BinaryTree.swift
//  
//
//  Created by EasyAiWithSwapnil on 26/08/25.
//

import Foundation
/*
class TreeNode {
    var val: Int
    var left: TreeNode?
    var right: TreeNode?
    init(_ val: Int) {
        self.val = val
    }
}

// Helper to ask user for input and return boolean
func promptBool(_ question: String) -> Bool {
    print("\(question) (yes/no): ", terminator: "")
    while let response = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        if response == "yes" || response == "y" { return true }
        if response == "no"  || response == "n" { return false }
        print("Please answer yes or no:")
    }
    return false
}

// Recursive function to build tree interactively
func buildTree(_ parentVal: Int?) -> TreeNode? {
    let prompt = parentVal == nil ? "Enter the value for the root:" : "Enter the value:"
    print(prompt, terminator: " ")
    guard let input = readLine(), let num = Int(input) else { return nil }
    let node = TreeNode(num)

    if promptBool("Do you want to enter on the left of \(num)?") {
        node.left = buildTree(num)
    }
    if promptBool("Do you want to enter on the right of \(num)?") {
        node.right = buildTree(num)
    }
    return node
}

// Print tree visually, level order
func printTree(_ root: TreeNode?) {
    guard let root = root else { return }
    var queue: [TreeNode?] = [root]
    while !queue.isEmpty {
        var nextQueue: [TreeNode?] = []
        var line = ""
        for node in queue {
            if let node = node {
                line += "\(node.val) "
                nextQueue.append(node.left)
                nextQueue.append(node.right)
            } else {
                // for visual: line += "- "
            }
        }
        if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { break }
        print(line)
        queue = nextQueue
    }
}


///// PRETTY PRINT FUNCTION /////
func prettyPrint(_ node: TreeNode?, _ prefix: String = "", _ isLeft: Bool = true) {
    guard let node = node else { return }
    if let right = node.right {
        prettyPrint(right, prefix + (isLeft ? "│   " : "    "), false)
    }
    print(prefix + (isLeft ? "└── " : "┌── ") + "\(node.val)")
    if let left = node.left {
        prettyPrint(left, prefix + (isLeft ? "    " : "│   "), true)
    }
}

// ---- MAIN -----
print("Let's build a binary tree!")
//let root = buildTree(nil)
//print("\nLevel order traversal of your tree:")
//printTree(root)
if let root = buildTree(nil) {
    print("\nPretty Print of your tree:")
    prettyPrint(root)
} else {
    print("No tree created.")
}
*/
class TreeNode {
    var val : Int
    var leftNode: TreeNode?
    var rightNode: TreeNode?
    init(_ val: Int){
        self.val = val
    }
}
    func promptBool(_ question: String) -> Bool {
        print("\(question) (yes/no): ", terminator:"")
        while let response = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            if response == "yes" || response == "y" { return true }
            if response == "no"  || response == "n" { return false }
            print("Please answer yes or no:")
        }
        return false
    }
    
    func buildTree(_ parentVal: Int?) -> TreeNode? {
        let prompt = parentVal == nil ? "Enter the value for the root:":"Enter the value:"
        
        print(prompt,terminator: " ")
        
        guard let input = readLine(), let num = Int(input) else { return nil }
        
        let node = TreeNode(num)
    
        if promptBool("Do you want to enter on left of \(num)"){
            node.leftNode = buildTree(num)
        }
        
        if promptBool("Do you want to enter on the right of \(num)?") {
            node.rightNode = buildTree(num)
        }
        
        return node
    }
    
    // level order using queue
    func printTree(_ root: TreeNode?){
        guard let root = root else {return}
        var queue: [TreeNode?] = []
        queue.append(root)
        while !queue.isEmpty {
            var size = queue.count
            var line = ""
            while size != 0 {
                let node = queue.removeFirst()
                if let node = node {
                    line += "\(node.val) "
                    queue.append(node.leftNode)
                    queue.append(node.rightNode)
                } else {
                    
                }
                size -= 1
            }
            if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { break }
            print(line)
        }
    }

print("Let's build a binary tree!")
let root = buildTree(nil)
print("\nLevel order traversal of your tree:")
printTree(root)
