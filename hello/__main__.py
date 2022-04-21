import click
from hello.src import there_command


@click.group()
@click.version_option(package_name="hello_template")
def main():
    """Hello is a CLI tool for creating a custom greeting to send to friend."""
    pass


# Note below that I am only using options, the click.argument can also be used but has limited capability.
# I generally like to use click.option and setting the option flag to required, as seen in the name option.
@click.option(
    # Notice that this has the is_flag option, this means that only the marker
    # will be used to trigger this option.
    "--greeting",
    "-g",
    is_flag=True,
    help="Adds a greeting to the final output",
)
@click.option(
    "--name",
    "-n",
    type=click.STRING,
    required=True,
    help="Adds the name to the final output.",
)
@main.command()
def there(name, greeting):
    """Create a Greeting to Send to a Friend!"""
    there_command.run(name, greeting)


if __name__ == "__main__":
    main()
