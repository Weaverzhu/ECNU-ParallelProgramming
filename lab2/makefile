main: main.c
	mkdir -p ./bin
	mpicc main.c -o main -std=c99 -O2
apimethod: apimethod.c
	mkdir -p ./bin
	mpicc apimethod.c -o api -Wall
clean:
	rm -f ./bin/*
tree: tree.c
	mkdir -p ./bin
	mpicc tree.c -o ./bin/tree -Wall
cross: cross.c
	mkdir -p ./bin
	mpicc cross.c -o ./bin/cross -Wall
