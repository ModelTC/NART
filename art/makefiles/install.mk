.PHONY: install

PREFIX ?= install
ART_ROOT ?= $(error ART_ROOT must me specified)
install: $(.DEFAULT_GOAL)
	@echo Installing...
	@mkdir -p $(PREFIX)
	@for i in $(INSTALLS); do \
		d=$(PREFIX)/`realpath --relative-base=$(BUILD_DIR) -s $$i`; \
		mkdir -p `dirname $$d` ;\
		cp $$i `dirname $$d` -rfL; \
	done;
