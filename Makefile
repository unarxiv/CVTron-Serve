default:
	@echo "Usage:"
	@echo "\tmake format

format:
	autoflake -i cvtron-serve/*.py
	autoflake -i cvtron-serve/**/*.py

	isort -i cvtron-serve/*.py
	isort -i cvtron-serve/**/*.py 

	yapf -i cvtron-serve/*.py
	yapf -i cvtron-serve/**/*.py
