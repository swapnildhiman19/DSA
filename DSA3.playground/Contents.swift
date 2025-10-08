import Foundation
/*
 class Solution {
 /*
  Actually simulating the doubling operation and recalculating GCD for each of the subArrays : O(N^3)
  func maxGCDScore(_ nums: [Int], _ k: Int) -> Int {
  /*
   
   For this question my though process of thinking was:
   
   1. A possible candidate could be maximum value present in the array, double it and return it, taking that as single subarray.
   
   2. I can see that in question N <= 1500 , which means N^2 operation is possible, so first thing first I'll fix the length of the subarray by using two for loop
   for i in (0...n) {
   for. j in (i...n) {
   // Current Subarray  = [i to j]
   
   Now for this subarray I will do the doubling operation , doubling the element which has least power of 2 in the current subarray would make the most sense as doubling an element is increasing the power of 2. Let's take an example to understand
   Current subarray : [2,4,3,6]
   Here power of 2 in 2 : 1
   power of 2 in 4 : 2
   power of 2 in 3 : 0
   power of 2 in 6 : 1
   Double the one which has least power so that it can contribute in GCD, in this case it would be 3 -> which becomes 6
   So new subarray becomes [2,4,6,6] now here GCD = 2 , len = 4 answer for this subarray would be 2*4 = 8
   
   Now let's see how k is dependent here:
   
   assume current subarray is
   [2,4,6,6] : k = 3 , double that element which has least power of 2
   [4,4,6,6] : GCD still remains 2
   [4,4,12,6] : GCD still remains 2
   [4,4,12,12] : GCD increases to 4
   
   Further I feel we would need a pre computed array which will store the what power of 2 is for each element , so that I don't do the operation of finding the best element to double again and again ( the one which has least power )
   
   We will do this for all the subarray and we will keep the best answer possible here.
   */
  
  //Helper to calculate gcd or hcf
  var gcd : (Int,Int) -> Int = { a,b in
  // Using Euclidean algorithm
  // gcd(a,0) = a
  // gcd(a,b) = gcd(b, a%b)
  // gcd(a,b,c) = gcd(gcd(a,b),c)
  var a = a , b = b
  while b != 0 {
  let temp = b
  b = a % b
  a = temp
  }
  return a
  }
  
  //Helper to calculate power of 2 in a number (number of trailing zeroes in the binary = 24 -> 2^3 x 3^1 : Binary form of 24 -> 11000
  var powerOf2 : (Int) -> Int = { x in
  var n = x
  var count = 0
  while n % 2 == 0 {
  n = n/2
  count += 1
  }
  return count
  }
  
  let n = nums.count
  // Precompute power of 2 for every number
  var powerOf2Arr = nums.map{ powerOf2($0) }
  
  var answer = 0
  
  //Trying for all the possible subarrays -> Fixing the length for each case
  for i in 0..<n {
  
  for j in i..<n {
  
  let len = j - i + 1
  var subArray = Array(nums[i...j])
  var subPower = Array(powerOf2Arr[i...j])
  
  // 1. Sort indices by power of 2 (ascending), so we can double those with least power first
  let indxs = subPower.enumerated().sorted { $0.element < $1.element }.map{ $0.offset }
  /*
   Given:
   nums = [12, 24, 32, 36]
   
   Step 1: Calculate power of 2 for each number
   12 = 2^2 * 3 → power = 2
   
   24 = 2^3 * 3 → power = 3
   
   32 = 2^5 → power = 5
   
   36 = 2^2 * 3^2 → power = 2
   
   So:
   
   swift
   subpowers = [2, 3, 5, 2]
   Each number's index is:
   
   12 (index 0), 24 (index 1), 32 (index 2), 36 (index 3)
   
   Step 2: Enumerate (attach index):
   This creates pairs:
   
   swift
   [(0, 2), (1, 3), (2, 5), (3, 2)]
   Meaning:
   
   (index 0, power 2)
   
   (index 1, power 3)
   
   (index 2, power 5)
   
   (index 3, power 2)
   
   Step 3: Sort by power value (ascending)
   Sort the array above by .element (the power):
   
   Sorted pairs:
   
   swift
   [(0, 2), (3, 2), (1, 3), (2, 5)]
   Both 12 (index 0) and 36 (index 3) have the lowest power (2)
   
   24 (index 1) is next with power 3
   
   32 (index 2) with power 5 is last
   
   Step 4: Extract only indices (map { $0.offset })
   Now, we just want the original ARRAY INDICES corresponding to these powers:
   
   swift
   [0, 3, 1, 2]
   0 is index of 12 (which has power 2)
   
   3 is index of 36 (power 2)
   
   1 is index of 24 (power 3)
   
   2 is index of 32 (power 5)
   */
  
  //How many can we double, can't exceed k
  let m = min(len, k)
  
  //Update the values and their power of 2, prioritizing the one with least power of 2 first
  for up in 0..<m {
  let idx = indxs[up]
  subArray[idx] *= 2
  subPower[idx] += 1
  }
  
  // Compute the GCD of modified subArray
  var g = subArray[0]
  for val in 1..<len {
  g = gcd(g,subArray[val])
  }
  answer = max(answer, len * g)
  }
  }
  return answer
  }
  */
 
 /*
  [2,4,6,6] : k = 7 , double that element which has least power of 2
  [4,4,6,6] : GCD still remains 2
  [4,4,12,6] : GCD still remains 2
  [4,4,12,12] : GCD increases to 4 // After this I am not allowed to double this mean for this subArray max GCD can be doubled by factor of 2 only , since each element can be doubled only once.
  
  
  ## **Layman’s Intuition**
  
  - When finding the best result for any subarray, **doubling the numbers whose “power of 2” is the lowest is best**—it’s these elements holding back the GCD.
  - *Why?* If you imagine the binary version, elements with fewer trailing zeros (“weak” in “2’s power”) limit the overall GCD.
  - By doubling (which adds one trailing zero), you bring the weakest elements closer to the rest.
  
  ***
  
  ## **Key Trick: `A[i] & -A[i]`**
  
  ### What does this bitwise expression do?
  - `A[i] & -A[i]` yields the **lowest set bit** of number A[i].
  - Example: If A[i] = 12 (1100₂), then `-A[i]` in binary is (0011₂)s complement → 0100₂. So 1100₂ & 0100₂ = 0100₂ = 4.
  - Mathematically, this is equivalent to $$2^{\text{power of two in A[i]}}$$
  
  - In code, it’s an efficient way to compute the **largest power of two dividing A[i]**.
  - This is **the same as your `powerOf2(x)` idea**, but much faster for this context (and more direct).
  
  ***
  
  ## **Why Only Consider Elements Where `minPow` is Minimal?**
  - For each subarray, the min power (smallest pow of 2 present) **controls the “potential” GCD** after k doubling operations.
  - Doubling all elements with this minPow will “tie them up” with the rest of the subarray or push the minPow higher—which can then boost the whole GCD.
  - **If you can double all numbers with minPow (and count ≤ k), you can increase the collective GCD by another factor of 2.**
  - **If count > k,** you simply can’t lift the “weakest link,” so the GCD can’t be increased.
  
  ***
  
  ## **Illustration with Example** (Your `, k = 3`):
  
  - First, calculate lowest power of 2 for each:
  - 2 = 2^1 × 1  → minPow: 2
  - 4 = 2^2 × 1  → minPow: 4
  - 6 = 2^1 × 3  → minPow: 2
  - 6 = 2^1 × 3  → minPow: 2
  
  - **minPow overall = 2**
  - There are 3 elements with minPow = 2 (at indices 0,2,3)
  
  - k=3 means you can double all the elements with minPow.
  - Double each one:
  - 2→4, 6→12, 6→12 (and 4 already has more trailing zeros)
  - Now all numbers have at least power of 2^2 (trailing two zeros).
  
  - GCD increases from 2 to 4!
  - **But if k = 2, you can’t boost GCD past 2, because at least one “weak link” remains.**
  
  ***
  
  ## **Detailed Step-by-Step (Why is this Faster)?**
  
  - For each subarray, **keep a running count of the minPow and how many numbers have it**.
  - If count (of minPow) ≤ k, GCD can be doubled.
  - There’s no need to simulate all k-choose-i double options; just count the weakest and see if you can double all in one go.
  
  ***
  
  ## **Bitwise Logic: `A[i] & -A[i]` Deep Dive**
  
  - In binary, `A[i]` has several bits; `-A[i]` is its two’s complement (invert then add 1).
  - Doing a bitwise & with `A[i]` and `-A[i]` **isolates the rightmost set bit**, i.e., the largest power of 2 that divides A[i].
  
  #### **Example:**
  - A[i] = 12:
  - binary = 1100
  - -A[i] = invert(1100) + 1 = 0011 + 0001 = 0100
  - 1100 & 0100 = 0100 = 4
  
  So the lowest set bit is 4, or in power terms, power of 2^2.
  
  
  ### **How is this different from your brute-force code?**
  - Instead of copying/sorting multiple times, you just keep track of:
  - The GCD so far (constant time using Euclidean)
  - The minPow so far (and a count)
  - When checking subarray [i...j]:
  - You only care about the numbers at minimal low bit (minPow): if there are **≤ k** such numbers, double all in one step, else not.
  - **Key:** You do NOT simulate/try all double arrangements—just count “weakest” power’s occurrences.
  
  ## **Walkthrough With Key Example (, k=3):**
  
  - subarray:
  - GCD after each step:
  - :    2, minPow=2, count=1, (count≤k→double): 2×2×1=4[1]
  - :  2, minPow=2, count=1+0=1, (count≤k): 2×2×2=8
  - : GCD=2; minPow(min(2,4,2))=2; count=1 (2) + 0 (4) + 1 (6)=2; (count≤k): 2×2×3=12
  - : GCD=2, minPow=2, count=3 (2,6,6). (k=3, can double all): after double, GCD becomes 4×4=16 (length=4). But GCD after double = 4, so score=4×4=16.
  
  **But, if you ever have more than k elements at minPow, you can't raise the GCD this way, so you just take current best!**
  
  ***
  
  ## **Summary Table**
  
  | Step           | What it Does                                  |
  |----------------|-----------------------------------------------|
  | A[i] & -A[i]   | Finds largest power of 2 dividing A[i]        |
  | Count minPow   | How many “weakest” in window                  |
  | If count ≤ k   | Can double all “weakest,” boost GCD by ×2     |
  | Else           | Take current GCD as is                        |
  | GCD update     | Rolling update with Euclidean algorithm       |
  | Score          | (GCD × (2/1)) × length                        |
  
  ***
  
  **This is the key to why the "accepted" code is far faster for large N—try only the genuinely impactful double operations instead of brute-forcing all possible double arrangements per subarray.**
  
  Let me know if you want more real-number examples, a dry-run, or further visual explanation!
  
  */
 
 // Helper for GCD
 func gcd(_ a: Int, _ b: Int) -> Int {
 var a = a, b = b
 while b != 0 {
 let t = b
 b = a % b
 a = t
 }
 return a
 }
 
 // Helper: Get lowest set bit (i.e., largest power of 2 dividing x)
 func lowBit(_ x: Int) -> Int { x & -x }
 
 func maxGCDScore(_ nums: [Int], _ k: Int) -> Int {
 let n = nums.count
 var answer = 0
 
 for i in 0..<n {
 var subGCD = nums[i]
 var minPow = lowBit(nums[i])
 var count = 1
 for j in i..<n {
 if j > i {
 subGCD = gcd(subGCD, nums[j])
 let lb = lowBit(nums[j])
 if lb < minPow {
 minPow = lb
 count = 1
 } else if lb == minPow {
 count += 1
 }
 }
 // Calculate the score: If we can double all minPow elements with at most k doubles.
 var curScore = subGCD * (count <= k ? 2 : 1) * (j - i + 1)
 answer = max(answer, curScore)
 }
 }
 return answer
 }
 }
 */

class TrieNode {
    var children : [Character:TrieNode] = [:]
}

class Solution {
    func longestCommonPrefix(_ strs: [String]) -> String {
        /*
         var smallestLengthStr = strs.min { a, b in
         a.count < b.count
         }
         
         var answer = smallestLengthStr
         
         for str in strs {
         if str == answer {
         continue
         }
         var currPrefixMatch = smallestLengthStr?.commonPrefix(with: str)
         if let currPrefixMatch = currPrefixMatch {
         if currPrefixMatch.count < answer!.count {
         answer = currPrefixMatch
         }
         } else {
         return ""
         }
         }
         
         return answer!
         */
        
        // Using Trie
        var smallestLengthString = strs.min { a, b in
            a.count < b.count
        }
        
        var smallestLengthStringCount = smallestLengthString!.count
        if strs.count == 1 {
            return strs[0]
        }
        
        if strs.contains("") {
            return ""
        }
        
        var root = TrieNode()
        
        //Building the Trie
        for str in strs {
            var node = root
            var strArray = Array(str)
            for i in 0..<smallestLengthStringCount {
                if node.children[strArray[i]] == nil {
                    node.children[strArray[i]] = TrieNode()
                }
                node = node.children[strArray[i]]!
            }
        }
        
        //Finding the first divergence in this single branch
        var answer = ""
        
        var node : TrieNode? = root
        /*
         // This is wrong not considering single leaf node case : "a"
         while (true && node != nil) {
         guard let children = node?.children else { break }
         if children.count > 1 {
         break
         }
         for (ch,childNode) in node!.children {
         answer += String(ch)
         node = childNode
         }
         }
         */
        
        while let currNode = node {
            let children = currNode.children
            
            if children.count > 1 || children.isEmpty {
                //Empty will be for leaf node
                break
            }
            
            //Since children.count == 1
            for (ch,childNode) in children {
                answer += String(ch)
                node = childNode
                break
            }
        }
        
        return answer
    }
    
    /*
     func romanToInt(_ s: String) -> Int {
     var i = 0
     var strArray = Array(s)
     let count = strArray.count
     
     var dictionary : [String:Int] = [
     "I"         :    1,
     "V"         :    5,
     "X"         :    10,
     "L"         :    50,
     "C"         :    100,
     "D"         :    500,
     "M"         :    1000
     ]
     
     var answer = 0
     
     while i < count {
     if i+1 < count {
     //Special cases
     if strArray[i] == "I" {
     if strArray[i+1] == "V" {
     answer += 4
     i += 2
     continue
     } else if strArray[i+1] == "X" {
     answer += 9
     i += 2
     continue
     }
     }
     
     if strArray[i] == "X" {
     if strArray[i+1] == "L" {
     answer += 40
     i += 2
     continue
     } else if strArray[i+1] == "C" {
     answer += 90
     i += 2
     continue
     }
     }
     
     if strArray[i] == "C" {
     if strArray[i+1] == "D" {
     answer += 400
     i += 2
     continue
     } else if strArray[i+1] == "M" {
     answer += 900
     i += 2
     continue
     }
     }
     }
     
     answer += dictionary[String(strArray[i])]!
     i += 1
     }
     return answer
     }
     */
    
    func romanToInt(_ s: String) -> Int {
        let values: [Character: Int] = [
            "I": 1, "V": 5, "X": 10,
            "L": 50, "C": 100, "D": 500, "M": 1000
        ]
        
        var result = 0
        let chars = Array(s)
        
        for i in 0..<chars.count {
            let value = values[chars[i]]!
            // If next value exists and is bigger, subtract current value
            if i+1 < chars.count, values[chars[i + 1]]! > value {
                result -= value
            } else {
                result += value
            }
        }
        return result
    }
    
    func intToRoman(_ num: Int) -> String {
        let roman: [(value: Int, symbol: String)] = [
            (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
            (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
        ]
        
        var num = num
        var result = ""
        for (value, symbol) in roman {
            while num >= value {
                result += symbol
                num -= value
            }
        }
        return result
        /*
         Using the descending order of list
         Example : 3350 -> M (2350) + M(1350) + M(350) + C(250) [Here can't use CM, D, CD] + C(150) + C(50) + L
         */
    }
    
    /*
     func longestPalindrome(_ s: String) -> String {
     var dpHashMap : [String:Bool] = [:]  //String key creation: O(log n) per operation due to string interpolation
     /*
      How the conversion works To convert an integer to its string representation, the system performs a sequence of divisions and modulo operations to extract each digit one by one. It divides the number by the base (10 for a decimal number) and takes the remainder to get the last digit.It repeats this process with the quotient until the number is reduced to zero.The extracted digits are then ordered correctly to form the final string. Since the number of digits in an integer is logarithmic with respect to its value (i.e., \(d\approx \log _{10}(n)\)), the complexity can also be expressed as \(O(\log n)\), where \(n\) is the value of the integer itself.
      */
     /*
      Thus need a 2D array, much better
      */
     var strArray = Array(s)
     var count = strArray.count
     
     var answer = String(strArray[0])
     var maxLen = 1
     
     for (i,ch) in strArray.enumerated() {
     dpHashMap["\(i)_\(i)"] = true
     }
     
     // Here the sequence of for loop would be for len in 1...count and then i in 0...count-len and not simply for i in 0...n-1 and for j in i...n-1
     because we would miss the cases , example
     "cbbd" here if you would have gone for later for loop pattern then you would try to deduce 0,3 without finding out 1,2 which would lead to wrong answer [IMP]
     
     for len in 1...count {
     
     for i in 0...count-len {
     
     var j = i + len
     
     if j >= count { break }
     
     var key = "\(i)_\(j)"
     
     if j == i {
     continue
     }
     
     if j - i == 1 {
     if strArray[j] == strArray[i] {
     // it's a palindrome
     dpHashMap[key] = true
     var newLen = j - i + 1
     if maxLen < newLen {
     maxLen = newLen
     answer = String(strArray[i...j])
     }
     }
     continue
     }
     // if i+1 ... j-1 is a palindrome and strArray[i] == strArray[j] that means i...j is a palindrome
     var subStringKey = "\(i+1)_\(j-1)"
     if let subStringPalindrome = dpHashMap[subStringKey] {
     if subStringPalindrome && strArray[i] == strArray[j] {
     dpHashMap[key] = true
     var newLen = j - i + 1
     if maxLen < newLen {
     maxLen = newLen
     answer = String(strArray[i...j])
     }
     }
     }
     }
     }
     
     return answer
     }
     */
    
    func longestPalindrome(_ s: String) -> String {
        
        let strArray = Array(s)
        let n = strArray.count
        
        var dp = Array(repeating: Array(repeating: false, count: n), count: n)
        
        var maxLen = 1
        var start = 0
        
        //Handle all the length
        for len in 1...n {
            for i in 0...n-len {
                var j = i + len - 1
                if len == 1 {
                    dp[i][j] = true //single character
                } else if len == 2 {
                    dp[i][j] = (strArray[i]==strArray[j])
                } else {
                    dp[i][j] = (strArray[i]==strArray[j]) && dp[i+1][j-1]
                }
                
                if dp[i][j] && maxLen < len {
                    maxLen = len
                    start = i
                }
            }
        }
        return String(strArray[start..<start+maxLen])
    }
    
    func isValid(_ s: String) -> Bool {
        
        var openingParanthesisStack : [Character] = []
        var strArray = Array(s)
        
        for character in strArray {
            if character == "(" || character == "{" || character == "[" {
                openingParanthesisStack.append(character)
            }
            
            if openingParanthesisStack.isEmpty {
                return false
            }
            
            if character == ")" {
                if openingParanthesisStack.removeLast() != "(" {
                    return false
                }
            }
            
            if character == "}" {
                if openingParanthesisStack.removeLast() != "{" {
                    return false
                }
            }
            
            if character == "]" {
                if openingParanthesisStack.removeLast() != "[" {
                    return false
                }
            }
        }
        
        return openingParanthesisStack.isEmpty ? true : false
    }
    
    func nextGreaterElement(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
        
        if nums2.isEmpty {
            return Array(repeating: -1, count: nums1.count)
        }
        
        var hashMapValToNextGreaterVal = [Int:Int]()
        
        var stackForNums2 = [Int]() //Need to find the next greater element for each of the element present in nums2
        
        stackForNums2.append(nums2[0])
        
        for element in nums2.dropFirst() {
            if !stackForNums2.isEmpty && element > stackForNums2.last! {
                //empty the stack and build the hashMap
                while !stackForNums2.isEmpty && element > stackForNums2.last! {
                    hashMapValToNextGreaterVal[stackForNums2.removeLast()] = element
                }
            }
            stackForNums2.append(element)
        }
        
        var answer = [Int]()
        
        print(hashMapValToNextGreaterVal)
        
        for element in nums1 {
            if let elementExist = hashMapValToNextGreaterVal[element] {
                answer.append(elementExist)
            } else {
                answer.append(-1)
            }
        }
        return answer
    }
    
    func sortStack(_ stack: [Int]) -> [Int] {
        /*
         Okay so seems like two recursive functions would be required here

         Example : [10,20,-5,7,15]

         1. pop out the element and do the recursive call on remaining of the stack to give the descending order stack: basically it will be
         [20,10,7,-5] + 15

         2. Another recursive function will be used now to insert the element in the already sorted stack we have using recursion. Basically we would say if sortedStack.top < element , simply pop the element and call recursive function on remaining sortedStack, after that once you found correct append our element and while exiting append back the previous pop element back to sortedStack.
         */
        var stack = stack
        sortStackHelper(&stack)
        return stack
    }
    
    func sortStackHelper(_ stack: inout [Int]) {
        // Base case: If the stack is empty or has 1 element, it's already sorted.
        if stack.isEmpty || stack.count == 1 {
            return
        }

        // Step 1: Pop the last element (top of stack)
        let temp = stack.removeLast()

        // Step 2: Sort the remaining stack recursively
        sortStackHelper(&stack)

        // Step 3: Insert the popped element back in the sorted stack
        insertStackHelper(&stack, temp)
    }

    func insertStackHelper(_ stack: inout [Int], _ element: Int) {
        // Base case: stack is empty or top of stack < element (for descending order)
        if stack.isEmpty || stack.last! < element {
            stack.append(element)
            return
        }

        // Pop the top and recurse
        let temp = stack.removeLast()
        insertStackHelper(&stack, element)
        // Push the previous top back
        stack.append(temp)
    }

    func nextSmallerElement(_ nums: [Int]) -> [Int] {
        
        if nums.isEmpty {
            return []
        }
        
        var stack : [(Int,Int)] = [] // stack to keep track of next smaller element
        
        stack.append((0,nums[0]))
        
        var output = Array(repeating: -1, count: nums.count)
        
        for index in 1..<nums.count {
            var element = nums[index]
            if !stack.isEmpty {
                var (stackLastIdx, stackLastElement) = stack.last!
                if stackLastElement < element {
                    stack.append((index,element))
                } else {
                    while !stack.isEmpty && stack.last!.1 > element {
                        var (currStackIdx, currStackElement) = stack.removeLast()
                        output[currStackIdx] = element
                    }
                    stack.append((index,element))
                }
            }
        }
        return output
    }
    
    /*
     Better approach is to store indices
     func nextSmallerElement(_ nums: [Int]) -> [Int] {
         var stack: [Int] = []
         var output = Array(repeating: -1, count: nums.count)
         for i in 0..<nums.count {
             while let last = stack.last, nums[last] > nums[i] {
                 output[last] = nums[i]
                 stack.removeLast()
             }
             stack.append(i)
         }
         return output
     }
     */
    
    //https://www.youtube.com/watch?v=wRubz1zhVqk
    //TimeComplexity = (log10 n)^2
    func findKthNumber(_ n: Int, _ k: Int) -> Int {
        var currPosition = 1
        var i = 1 // when i == k that means we have reached our desired currPosition : we are basically trying to simulate the DFS PreOrder without actually doing it completely
        while i < k {
            let steps = countTotalChildrean(n, currPosition) // This will give the number of children this subTree including the root has
            if i + steps <= k {
                // move towards right
                i += steps
                currPosition += 1
            } else {
                // move deep down
                i += 1
                currPosition *= 10
            }
        }
        return currPosition
    }
    
    func countTotalChildrean(_ n: Int, _ curr1: Int) -> Int {
        var count = 0
        var curr1 = curr1
        var curr2 = curr1 + 1
        
        // will be counting level by level
        while curr1 <= n {
            count += min(n+1, curr2) - curr1 //at this level
            curr1 *= 10
            curr2 *= 10
        }
        
        return count
    }
    
    
    
    func lexicalOrder(_ n: Int) -> [Int] {
        // first digit has option from 0...9 , second digit will have option from 1...9
        // DFS preorder traversal will give the answer
        var output = [Int]()
        for i in 1...9 {
            dfs(i,n,&output)
        }
        return output
    }

    func dfs(_ curr: Int, _ n :Int, _ output: inout [Int]) {
        if curr > n {
            return
        }
        output.append(curr)
        for i in 0...9 {
            var newElement  = (curr*10) + i
            dfs(newElement, n, &output)
        }
    }
    
    
    func threeSum(_ nums: [Int]) -> [[Int]] {
            var nums = nums.sorted()
            return kSum(nums,0,0,3)
        }
        func kSum(_ nums: [Int], _ start: Int, _ target: Int, _ k :Int) -> [[Int]] {
            let n = nums.count
            var result = [[Int]]()
            if k == 2 {
                var start = start
                var end = n - 1
                while start < end {
                    if nums[start] + nums[end] == target {
                        result.append([start,end])
                        start += 1
                        end -= 1
                        while start < end && nums[start] == nums[start+1] {
                            start += 1
                        }
                        while end > start && nums[end] == nums[end-1] {
                            end -= 1
                        }
                    } else if nums[start] + nums[end] < target {
                        start += 1
                    } else {
                        end -= 1
                    }
                }
                return result
            }
            var i = start
            while i < n - k + 1 {
                //Same i needs to be skipped to remove duplicates
                if i > start && nums[i] == nums[i-1]  {
                    i += 1
                    continue
                }

                var subResults = kSum(nums, i + 1, target - nums[i], k - 1)
                
                for subResult in subResults {
                    result.append([nums[i]]+subResult)
                }
                
                i += 1
            }
            return result
        }
    
        func convert(_ s: String, _ numRows: Int) -> String {
            if numRows == 1 || numRows > s.count {
                return s
            }
            
            var stringAtEachRow : [String] = Array(repeating: "", count: numRows)
            
            var strArray = Array(s)
            
            var index = 0
            
            while index < strArray.count {
                //Go down and then diagonally up
                var currRow = 0
                
                while currRow <= numRows - 1 && index < strArray.count {
                    stringAtEachRow[currRow].append(strArray[index])
                    currRow += 1
                    index += 1
                }
                
                currRow -= 2
                
                while currRow > 0 && index < strArray.count {
                    stringAtEachRow[currRow].append(strArray[index])
                    currRow -= 1
                    index += 1
                }
            }
            
            var output : String = ""
            for eachRowString in stringAtEachRow {
                output.append(eachRowString)
            }
            
            return output
        }
    
    func findPeakElement(_ nums: [Int]) -> Int {
        let n = nums.count

        var left = 0
        var right = n - 1

        while left < right {
            let mid = left + (right - left)/2
            if mid > 0 && mid < n && nums[mid-1] < nums[mid] && nums[mid] < nums[mid+1] {
                return mid //found the peek
            }
            if nums[mid] < nums[mid+1] {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return left
    }
    

    
}


let solution = Solution()
//print(solution.nextGreaterElement([1,3,5,2,4], [6,5,4,3,2,1,7]))
//print(solution.sortStack([10,20,-5,7]))
//print(solution.nextSmallerElement([10,9,8,7]))
//print(solution.findKthNumber(838, 500))
//print(solution.lexicalOrder(13))
//print(solution.threeSum([-1,0,1,2,-1,-4]))
print(solution.convert("ABCDE", 4))
