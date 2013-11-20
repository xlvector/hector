package main

import(
	"reflect"
	"fmt"
)

type A interface {
	f(a int) int
}

type B struct {
	c int
}

func (self *B) f(a int) int {
	return a + self.c
}

func main(){
	var x A
	x = &(B{3})
	fmt.Println(reflect.TypeOf(x))
}