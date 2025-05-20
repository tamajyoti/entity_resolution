import click


class OptionThatRequiresOthers(click.Option):

    """Option that demands other arguments to be given depending on the value of the argument."""

    def __init__(self, *args, **kwargs):
        self.required_params = kwargs.pop("required_params") if "required_params" in kwargs else {}
        if self.required_params is None:
            self.required_params = {}
        additional_help = []
        for this_value, other_params in self.required_params.items():
            additional_help.append(f"{this_value}: {','.join(other_params)}")
        kwargs["help"] = (
            kwargs.get("help", "")
            + "\n Note: depending on values of this argument, other arguments required:"
            + "\n".join(additional_help)
        ).strip()
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        """Check if certain argument are given depending on the value of the argument."""
        for param in self.required_params[opts[self.name]]:
            if not ((param in opts and opts[param]) or (param in ctx.params)):
                ctx.fail(f"Argument {param} is required when {self.name}=={opts[self.name]}")
        return super().handle_parse_result(ctx, opts, args)
