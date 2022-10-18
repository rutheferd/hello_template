import click
from hello.src import there_command


@click.group()
@click.version_option(package_name="hello_template")
def main():
    """This is our new tool that will handle the stonesoup data generation."""
    pass


# Note below that I am only using options, the click.argument can also be used but has limited capability.
# I generally like to use click.option and setting the option flag to required, as seen in the name option.
@click.option(
    # Notice that this has the is_flag option, this means that only the marker
    # will be used to trigger this option.
    "--plot",
    "-p",
    is_flag=True,
    help="Plots the final output",
)
@click.option(
    "--seed",
    "-s",
    type=click.INT,
    required=True,
    help="The seed for the generation randomization",
)
@main.command()
def generate(seed, plot):
    """Create a Greeting to Send to a Friend!"""
    there_command.run(seed, plot)


if __name__ == "__main__":
    main()
