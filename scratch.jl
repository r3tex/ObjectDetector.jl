module Foo

module Bar
thing(x::String) = 1
end
import .Bar: thing

module Baz
thing(x::Int) = 1
end
import .Baz: thing

export thing
end
