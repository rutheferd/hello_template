from click.testing import CliRunner
from hello.src import there_command
from hello.__main__ import main
import pytest

def test_there_function():
    name = "Austin"
    greeting = True
    assert (
        there_command.there(name, greeting)
        == "Hello there Austin, how are you?"
    )

    name = "Austin"
    assert there_command.there(name) == "Hello there Austin."

    name = "Austin"
    greeting = False
    assert there_command.there(name, greeting) == "Hello there Austin."

    pass


def test_there_command():
    # Testing with greeting
    name = "Austin"
    runner = CliRunner()
    result = runner.invoke(main, ["there", "-n", name, "-g"])
    assert result.output == "Hello there Austin, how are you?\n"

    # Testing without greeting
    name = "Obi-Wan"
    result = runner.invoke(main, ["there", "-n", name])
    assert result.output == "Hello there Obi-Wan.\n"

    pass


def test_training_command():
        file = ""
        runner = CliRunner()
        #Test exception thrown when no file inputted
        with pytest.raises(FileNotFoundError):
            result = runner.invoke(main, ["-te"])
        with pytest.raises(ValueError):
            result = runner.invoke(main, ["some_file.png"])


