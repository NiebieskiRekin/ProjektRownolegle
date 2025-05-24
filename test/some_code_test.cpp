#include <gtest/gtest.h>
// #include "src/some_code.h"

TEST(SomeCodeTest, SomeFunctionPrints) {
    // Capture stdout for testing output
    std::stringstream ss;
    std::streambuf* oldCout = std::cout.rdbuf();
    std::cout.rdbuf(ss.rdbuf());

    // someFunction();

    std::cout.rdbuf(oldCout); // Restore cout

    ASSERT_NE(ss.str().find("This is some code."), std::string::npos);
}