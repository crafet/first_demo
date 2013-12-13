/**
  main test 
  @author liuyilin
  @date 2013-12-12
*/

#include "hello.h"

int main()
{
	std::cout << "enter main ..." << std::endl;
	test::crafet::Hello hello("crafet");
	hello.say_hello();
	std::cout << "exit main ..." << std::endl;
	return 0;

}
