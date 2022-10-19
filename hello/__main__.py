from email.policy import default
import click
from hello.src import simulate_command, there_command, kalman_command


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
    pass


@click.option(
    # Notice that this has the is_flag option, this means that only the marker
    # will be used to trigger this option.
    "--truths_path",
    "-t",
    type=click.STRING,
    required=True,
    help="Plots the final output",
)
@click.option(
    "--save_path",
    "-s",
    default="./sim.txt",
    type=click.STRING,
    help="The seed for the generation randomization",
)
@main.command()
def simulate(truths_path, save_path):
    """Create a Greeting to Send to a Friend!"""
    simulate_command.run(truths_path=truths_path, save_path=save_path)
    pass


@main.command()
def track():
    """Create a Greeting to Send to a Friend!"""
    kalman_command.run()
    pass


if __name__ == "__main__":
    main()
