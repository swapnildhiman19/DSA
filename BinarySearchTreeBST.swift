//
//  BinarySearchTreeBST.swift
//  
//
//  Created by EasyAiWithSwapnil on 26/08/25.
//

import Foundation

class BSTNode {
    var val : Int
    var left : BSTNode?
    var right : BSTNode?
    init(_ value: Int){
        self.val = value
    }
}

class BST {
    private(set) var root : BSTNode? //can publicly read or access this property but only private access to write this variable
    
    //insert
    func insert(_ value : Int) {
        root = insertHelper(root, value)
    }
    
    func insertHelper(_ node: BSTNode?, _ val: Int) -> BSTNode? {
        guard let node = node else { return BSTNode(val) }
        if val == node.val { return node }
        if val < node.val {
            // traverse left
            node.left = insertHelper(node.left, val) // This will return the same node if it's present and if not it will return the new node and we need to append that value to correct place in the leaf level
            /*
             You only change the left or right child, NOT the node itself, unless you’re at the very root or deleting/creating a node.
             Always assign to .left or .right!
             */
        } else {
            // traverse right
            node.right = insertHelper(node.right , val)
        }
        return node
    }
    
    func search(_ value: Int) -> BSTNode? {
        var current = root
        while let node = current {
            if node.val == value { return node }
//            else if node.val < value {
//                current = node.right
//            } else {
//                current = node.left
//            }
            current = value < node.val ? node.left : node.right
        }
        return nil
    }
    
    //Delete :
    /*
     three cases in delete:
     1. Simple leaf node will be getting deleted : No issue
     2. Node has only one child , delete the node and attach the deleted node child to it's parent
     3. Both child are present : Find the next inorder successor of deleted node ( after that replace the value of with that node and delete that node also ) : inorder -> left root right
     */
    func delete(_ value : Int){
        root = deleteHelper(root, value)
    }
    
    //returns the (potentially new) subtree/root node after deletion, allowing parent pointers to be properly updated or replaced.
    func deleteHelper(_ node: BSTNode?, _ val: Int) -> BSTNode? {
        
        guard let node = node else { return nil }
        
        if val < node.val {
            node.left = deleteHelper(node.left, val)
        } else if val > node.val {
            node.right = deleteHelper(node.right, val)
        } else {
            if node.left == nil { return node.right }
            
            if node.right == nil { return node.left }
            
            // try to replace it with next inorder successor - try to find the next biggest element here, i.e. traverse to all the left in node.right
            if let minRight = minNode(node.right) {
                node.val = minRight.val
                node.right = deleteHelper(node.right, minRight.val) // delete that bottom successor node
            }
        }
        
        return node
    }
    
    func minNode(_ node: BSTNode?) -> BSTNode? {
        guard var node = node else { return nil }
        while let next = node.left { node = next }
        return node
    }
    
    //checking isBalanced
    func isBalanced() -> Bool {
        func check(_ node: BSTNode?) -> (Bool, Int) {
            guard let node = node else { return (true, 0)}
            let (leftBal,leftH) = check(node.left)
            let (rightBal,rightH) = check(node.right)
            let balanced = leftBal && rightBal && abs(leftH-rightH) <= 1
            return (balanced, 1+max(leftH,rightH))
        }
        return check(root).0
    }
    
    func printLevels(){
        guard let root = root else {
            print("No BST existence")
            return
        }
        var queue : [BSTNode?] = []
        queue.append(root)
        
        while !queue.isEmpty {
            var size = queue.count
            while size != 0 {
                guard let node = queue.removeFirst() else { continue }
                print(node.val, terminator: " ")
                if let l = node.left {queue.append(l)}
                if let r = node.right {queue.append(r)}
                size -= 1
            }
            print() // adds a new line
        }
    }
}

let bst = BST()
[15, 10, 20, 5, 12, 11].forEach { bst.insert($0) }
print("Tree (levels):")
bst.printLevels()
print("Is balanced? \(bst.isBalanced())")
print("Searching for 12:", bst.search(12) != nil)
print("Deleting 10...")
bst.delete(10)
bst.printLevels()
print("Is balanced? \(bst.isBalanced())")
