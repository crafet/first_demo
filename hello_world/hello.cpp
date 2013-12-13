/**
  impletation for class Hello

*/

#include "hello.h"

namespace test {
namespace crafet {

Hello::Hello(const char* str)
{
	string = str;
}

void Hello::say_hello()
{
	std::cout << "Hello, " << string << std::endl;
}

} // end of namespace crafet
} // end of namespace test
