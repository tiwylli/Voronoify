
# Top-level Makefile delegates to component-level build files.
all:
	@$(MAKE) -C cuda

clean:
	@$(MAKE) -C cuda clean
