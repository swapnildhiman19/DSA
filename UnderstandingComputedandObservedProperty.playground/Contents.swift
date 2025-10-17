import Foundation

var width : Float = 3.4
var height : Float = 2.1

var areaCovered: Float {
    get {
        return height * width
    }
}

var areaCoveredByOneBucket : Float = 1.5

var bucketsOfPaint : Int {
    get {
        return Int(ceil(areaCovered/areaCoveredByOneBucket))
    }
    set {
        print("This amount of paint can cover the area of \(Float(newValue) * areaCoveredByOneBucket)")
    }
}

print(bucketsOfPaint)

bucketsOfPaint = 8
