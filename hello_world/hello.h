/*
   hello class declaretion

*/

#ifndef HELLO_H
#define HELLO_H

#include <iostream>

namespace test {
namespace crafet {

class Hello
{
	public:
		Hello(const char* str);
		void say_hello();

	private:
		const char* string;
};

} // end of namesapce crafet
} // end of namespace test
#endif
