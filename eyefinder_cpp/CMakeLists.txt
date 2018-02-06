#
#  _______ _    _ _____  _____     _____  _____
# |__   __| |  | |_   _|/ ____|   |_   _|/ ____|       /\
#    | |  | |__| | | | | (___       | | | (___        /  \
#    | |  |  __  | | |  \___ \      | |  \___ \      / /\ \
#    | |  | |  | |_| |_ ____) |    _| |_ ____) |    / ____ \
#    |_|__|_|_ |_|_____|_____/__  |_____|_____/    /_/  _ \_\
#   |__   __| |  | |__   __/ __ \|  __ \|_   _|   /\   | |
#      | |  | |  | |  | | | |  | | |__) | | |    /  \  | |
#      | |  | |  | |  | | | |  | |  _  /  | |   / /\ \ | |
#      | |  | |__| |  | | | |__| | | \ \ _| |_ / ____ \| |____
#      |_|   \____/   |_|  \____/|_|  \_\_____/_/    \_\______|
#
#
#    _____  ______          _____      _______ _    _ ______
#   |  __ \|  ____|   /\   |  __ \    |__   __| |  | |  ____|
#   | |__) | |__     /  \  | |  | |      | |  | |__| | |__
#   |  _  /|  __|   / /\ \ | |  | |      | |  |  __  |  __|
#   | | \ \| |____ / ____ \| |__| |      | |  | |  | | |____
#   |_|__\_\______/_/_ __\_\_____/__ _   |_|__|_|_ |_|______|_ _ _
#  / ____/ __ \|  \/  |  \/  |  ____| \ | |__   __/ ____| | | | | |
# | |   | |  | | \  / | \  / | |__  |  \| |  | | | (___   | | | | |
# | |   | |  | | |\/| | |\/| |  __| | . ` |  | |  \___ \  | | | | |
# | |___| |__| | |  | | |  | | |____| |\  |  | |  ____) | |_|_|_|_|
#  \_____\____/|_|  |_|_|  |_|______|_| \_|  |_| |_____/  (_|_|_|_)
#
#
#
# This is a CMake makefile.  CMake is a tool that helps you build C++ programs.
# You can download CMake from http://www.cmake.org.  This CMakeLists.txt file
# you are reading builds dlib's example programs.
#


cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(examples)


# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# cmake needs is the dlib source code folder and it will take care of everything.
add_subdirectory(../dlib dlib_build)


# The next thing we need to do is tell CMake about the code you want to
# compile.  We do this with the add_executable() statement which takes the name
# of the output executable and then a list of .cpp files to compile.  Here we
# are going to compile one of the dlib example programs which has only one .cpp
# file, assignment_learning_ex.cpp.  If your program consisted of multiple .cpp
# files you would simply list them here in the add_executable() statement.
#add_executable(assignment_learning_ex assignment_learning_ex.cpp)
# Finally, you need to tell CMake that this program, assignment_learning_ex,
# depends on dlib.  You do that with this statement:
#target_link_libraries(assignment_learning_ex dlib::dlib)


if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build the eyefinder example.")
else()
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})

      add_executable(eyefinder main.cpp eyefinder.cpp)
      target_link_libraries(eyefinder dlib::dlib ${OpenCV_LIBS} )
   else()
      message("OpenCV not found, so we won't build the eyefinder example.")
   endif()
endif()
