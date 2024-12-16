##
## EPITECH PROJECT, 2024
## Makefile
## File description:
## Makefile
##

PATH_ANALYZER	=	analyzer/
PATH_GENERATOR	=	generator/

all:
	cd $(PATH_ANALYZER) && $(MAKE)
	cd $(PATH_GENERATOR) && $(MAKE)

clean:
	cd $(PATH_ANALYZER) && $(MAKE) clean
	cd $(PATH_GENERATOR) && $(MAKE) clean
	rm -f *.gcno *.gcda gmon.out

fclean: clean
	cd $(PATH_ANALYZER) && $(MAKE) fclean
	cd $(PATH_GENERATOR) && $(MAKE) fclean

re:
	cd $(PATH_ANALYZER) && $(MAKE) re
	cd $(PATH_GENERATOR) && $(MAKE) re

unit_tests:
	cd $(PATH_ANALYZER) && $(MAKE) unit_tests
	cd $(PATH_GENERATOR) && $(MAKE) unit_tests

tests_run:
	cd $(PATH_ANALYZER) && $(MAKE) tests_run
	cd $(PATH_GENERATOR) && $(MAKE) tests_run

gcovr_analyzer:
	cd $(PATH_ANALYZER) && $(MAKE) gcovr

gcovr_generator:
	cd $(PATH_GENERATOR) && $(MAKE) gcovr

debug_analyzer:
	cd $(PATH_ANALYZER) && $(MAKE) debug

debug_generator:
	cd $(PATH_GENERATOR) && $(MAKE) debug

profiling_analyzer:
	cd $(PATH_ANALYZER) && $(MAKE) debug

profiling_generator:
	cd $(PATH_GENERATOR) && $(MAKE) debug

compilation_database: clean
	bear -- make

code_analyse: compilation_database
	clang-tidy --config-file=.clang-tidy \
		-header-filter=.* --use-color $(SRC)

.PHONY: all clean fclean re unit_tests tests_run gcovr_analyzer \
	gcovr_generator debug_analyzer debug_generator profiling_analyzer \
	profiling_generator compilation_database code_analyse
