
//MARK: Strong Reference Understanding
/*
 1 Strong Reference
 
class Bird {
    var nest: Nest? // Strong reference
    deinit { print("Bird deallocated") }
}

class Nest {
    var bird: Bird? // Strong reference
    deinit { print("Nest deallocated") }
}

// ARCbird = 0, ARCnest = 0

var b: Bird? = Bird()         // ARCbird = 1 (b points to Bird instance)
var n: Nest? = Nest()         // ARCnest = 1 (n points to Nest instance)

b!.nest = n                   // ARCnest = 2 (n, b!.nest both point to Nest)
n!.bird = b                   // ARCbird = 2 (b, n!.bird both point to Bird)

b = nil                       // ARCbird = 1 (still held by n!.bird), ARCnest = 2
n = nil                       // ARCnest = 1 (still held by b!.nest), ARCbird = 1

// Both ARCbird and ARCnest stay at 1: no deinit is called, memory is leaked! -> Loop is getting created


class Bird {
    var nest: Nest? // strong reference
    deinit { print("Bird deallocated") }
}

class Nest {
    var bird: Bird? // strong reference
    deinit { print("Nest deallocated") }
}

// ARCbird = 0, ARCnest = 0

var b: Bird? = Bird()          // ARCbird = 1
var n: Nest? = Nest()          // ARCnest = 1

b!.nest = n                    // ARCnest = 2 (b!.nest holds n), ARCbird = 1
n!.bird = b                    // ARCbird = 2 (n!.bird holds b), ARCnest = 2

b = nil                        // ARCbird = 1 (only n!.bird holds b), ARCnest = 2
n!.bird = b                    // n!.bird = nil; ARCbird = 0 (no refs to b anymore), ARCnest = 2

n = nil                        // ARCnest = 1 (only b!.nest holds n, but b is now nil), so now ARCnest = 0

// Now both ARCbird and ARCnest are 0, both are deallocated; you see:
// Bird deallocated
// Nest deallocated

By setting n!.bird = b (when b is already nil), you set nest's bird to nil, removing the last strong reference to Bird, so Bird is deallocated immediately. The only thing holding Nest was b!.nest, but since b is nil, nothing points to Nest now—so Nest deallocates.

class Bird {
    var nest: Nest? // strong reference
    deinit { print("Bird deallocated") }
}

class Nest {
    var bird: Bird? // strong reference
    deinit { print("Nest deallocated") }
}

// ARCbird = 0, ARCnest = 0

var b: Bird? = Bird()          // ARCbird = 1 (b -> Bird1)
var n: Nest? = Nest()          // ARCnest = 1

b!.nest = n                    // ARCnest = 2 (b!.nest -> n), ARCbird = 1
n!.bird = b                    // ARCbird = 2 (n!.bird -> Bird1), ARCnest = 2

b = nil                        // ARCbird = 1 (still n!.bird -> Bird1), ARCnest = 2

n!.bird = Bird()               //
// (1) n!.bird's old value (Bird1) ARCbird = 0 (deinit)
//     print("Bird deallocated")
// (2) n!.bird now points to Bird2. ARCbird(Bird2) = 1 (from n!.bird)

n = nil                        // ARCnest = 1 (b!.nest -> n, but b = nil so gone, ARCnest = 0, deinit)
// n!.bird releases Bird2, ARCbird(Bird2) = 0, Bird2 deinit

// You see:
// Bird deallocated   <-- (from releasing Bird1)
// Nest deallocated
// Bird deallocated   <-- (from releasing Bird2)

 What's happening?
 n!.bird = Bird() breaks the strong reference to the original Bird object (held by Nest), so Bird1 deallocates first.

 The new temporary Bird object (Bird2) is only referenced by n!.bird.

 After n = nil, Nest deallocates, and now Bird2 also deallocates because nothing holds it anymore.

 Key ARC Rule Shown:
 Assigning a new object to a strong reference property first releases the old object (possibly triggering deinit if its count drops to zero) and then increments the count of the new object.

 Setting it to nil (n!.bird = b, b == nil) releases the reference from n!.bird, so Bird reference count drops to zero → deinit.
*/


//MARK: Weak Reference
/*
class Bird {
    var nest: Nest? // Strong reference
    deinit { print("Bird deallocated") }
}

class Nest {
    weak var bird: Bird? // Weak reference (does not increase ARCbird)
    deinit { print("Nest deallocated") }
}

// ARCbird = 0, ARCnest = 0

var b: Bird? = Bird()         // ARCbird = 1 (b points to Bird instance)
var n: Nest? = Nest()         // ARCnest = 1 (n points to Nest instance)

b!.nest = n                   // ARCnest = 2 (n, b!.nest both point to Nest)
n!.bird = b                   // ARCbird = 1 (still just 'b') (bird is weak, does NOT increase ARC)

b = nil                       // ARCbird = 0 (no strong refs, Bird deallocated, n!.bird is now nil)
                              // ARCnest = 2 (n, b!.nest since Bird out of memory, b!.nest is gone)
n = nil                       // ARCnest = 1->0 (now no strong refs, Nest deallocated)

// Both classes' deinit print: NO MEMORY LEAK!

*/


