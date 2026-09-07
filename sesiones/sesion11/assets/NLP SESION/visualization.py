from IPython.display import HTML, display
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Union


class TokenVisualization:
    """ A class to visualize token boundaries as text

    Attributes:
        encoder (Callable): A callable function to encode the text into tokens.
        decoder (Callable): A callable function to decode tokens back into text.
        cmap (str, optional): The colormap to use for token colors.
        font_family (str, optional): The font family to use for the tokens.
        transparency (float, optional): The transparency level for the token background colors.
        font_size (str, optional): The font size to use for the tokens.
        unk_token (str, optional): The string representation for unknown tokens.
        font_weight (int, optional): The font weight to use for the tokens.
        padding (str, optional): The padding to use for the tokens.
        margin_right (str, optional): The right margin to use for the tokens.
        border_radius (str, optional): The border radius to use for the tokens.
        background_color (str, optional): The background color for the container element.


        ##################################################################
        ########################## EXAMPLE USAGE #########################
        ##################################################################

            # Initialize the TokenVisualization class with the encoder and decoder functions
            token_viz = TokenVisualization(
                encoder=lambda x: list(x),
                decoder=lambda x: x,
            )

            # Define a sample text to visualize tokenization boundaries
            sample_text = "This is a sample text.\nIt has multiple lines."

            # Visualize the tokenization boundaries
            html = token_viz.visualize(sample_text)
            display(HTML(html))

        ##################################################################
    """
    def __init__(self, encoder: Callable, decoder: Callable, cmap: str = 'Pastel2', font_family: str = 'Courier New',
                 transparency: float = 0.675, font_size: str = '1.1em', unk_token: str = '???', font_weight: int = 300,
                 padding: str = '0px', margin_right: str = '0px', border_radius: str = '0px',
                 background_color: str = '#F0F0F0', font_color: str = 'black'):
        self.encoder = encoder
        self.decoder = decoder
        self.cmap = cmap
        self.font_family = font_family
        self.transparency = transparency
        self.font_size = font_size
        self.unk_token = unk_token
        self.font_weight = font_weight
        self.padding = padding
        self.margin_right = margin_right
        self.border_radius = border_radius
        self.background_color = background_color
        self.font_color = font_color

    def get_color(self, value: int) -> str:
        """ Generates a CSS background-color string using the specified colormap and value.

        Args:
            value (int): The value to use for generating the background color.

        Returns:
            str: A CSS background-color string.
        """
        colormap = plt.get_cmap(self.cmap)
        return f"background-color: rgba{tuple([int(x * 255) for x in colormap(value % colormap.N)[:-1]] + [self.transparency, ])};"

    @staticmethod
    def replace_rightmost_newline(s: str, replacement: str = '<br>') -> str:
        """ Replaces the rightmost newline character with the specified replacement.

        Args:
            s (str): The input string containing newline characters.
            replacement (str, optional): The string to replace the rightmost newline with.

        Returns:
            str: The modified string with the rightmost newline replaced.
        """
        parts = s.rsplit('\\n', 1)
        return replacement.join(parts)

    def generate_style(self) -> str:
        """ Generates a CSS style string for token elements.

        Returns:
            str: A CSS style string for token elements.
        """
        style = f"<style>span.token {{font-family: {self.font_family}; font-size: {self.font_size}; " \
        f"font-weight: {self.font_weight}; padding: {self.padding}; margin-right: {self.margin_right}; " \
        f"border-color: rgba(0, 0, 0, 0.05); border-style: ridge; border-radius: {self.border_radius}; " \
        f"color: {self.font_color};}}</style>"
        return style

    def generate_tokens(self, token_lines: List[List[str]], strip_padding: bool, pad_token: Union[int, str] = 0) -> str:
        """ Generates an HTML string containing the decoded token strings with their styles.

        Args:
            token_lines (List[List[str]]): A list of lists of tokens to generate HTML for.
            strip_padding (bool): If True, remove padding tokens from the visualization.
            pad_token (Union[int, str], optional): The value of the padding token (default: 0).

        Returns:
            str: An HTML string containing the styled tokens.
        """
        tokens_html = ""
        for token_line in token_lines:
            current_tokens = []

            for i, token in enumerate(token_line):
                if strip_padding and token == pad_token:
                    continue

                current_tokens.append(token)
                decoded_token = self.decoder(current_tokens)
                color = self.get_color(i)

                # For BPE tokenizers and uncompleted merges we need to continue until
                # the tokenizer decodes a valid token.
                if "�" in decoded_token:
                    continue

                try:
                    token_html = f"""
                    <div style="display: inline-block; text-align: center; margin-right: {self.margin_right};">
                        <span class='token' style='{color} padding: 4px 6px; border-radius: 5px;'>{decoded_token.replace(' ', '&nbsp;')}</span><br>
                        <span style="font-size: 0.9em; color: #333;">{token}</span>
                    </div>
                    """
                    tokens_html += token_html
                    # tokens_html += f"<span class='token' style='{color}'>{decoded_token.replace(' ', '&nbsp;')}</span>".replace(
                    #     '\t', '\\t').replace('\n', '\\n').replace('\r', '\\r').replace('\f', '\\f').replace('\v', '\\v')
                except TypeError:
                    tokens_html += f"<span class='token' style='{color}'>{self.unk_token}</span>"

                current_tokens = []
            tokens_html = self.replace_rightmost_newline(tokens_html)
        return tokens_html

    def generate_container(self,
                           token_lines: List[List[str]],
                           strip_padding: bool,
                           pad_token: Union[int, str] = 0) -> str:
        """ Generates an HTML container string with the specified token_lines.

        Args:
            token_lines (List[List[str]]): A list of lists of tokens to generate the container for.
            strip_padding (bool): If True, remove padding tokens from the visualization.
            pad_token (Union[int, str], optional): The value of the padding token (default: 0).

        Returns:
            str: An HTML container string containing the token_lines.
        """
        container_html = f"<div style='background-color: {self.background_color}; line-height: 175%; padding: 25px; " \
                         f"border-radius: 8px; margin-left: 10px; margin-right: 10px; margin-top: 20px; " \
                         f"margin-bottom: 20px; overflow-x: auto; white-space: nowrap;'>"
        container_html += self.generate_tokens(token_lines, strip_padding, pad_token)
        container_html += "</div>"
        return container_html

    def visualize(self, text: Union[str, list], split_on: str = "\n", display_inline: bool = False,
                  encoder: Optional[Callable] = None, decoder: Optional[Callable] = None,
                  strip_padding: bool = True, pad_token: Union[int, str] = 0) -> str:
        """ Visualizes the given text by generating an HTML string with styled tokens.

        Args:
            text (str, list): The text to visualize. (can be a list of tokens or a string)
            split_on (str): The delimiter used to split the text into separate lines (default: newline character).
            display_inline (bool): Whether to display the generated HTML inline using IPython display (default: False).
            encoder (Optional[Callable]): An optional encoder function to override the default encoder (default: None).
            decoder (Optional[Callable]): An optional decoder function to override the default decoder (default: None).
            strip_padding (bool, optional): If True, remove padding tokens from the visualization (default: True).
            pad_token (Union[int, str], optional): The value of the padding token (default: 0).

        Returns:
            str: The generated HTML string containing the styled tokens and their container.
        """

        _encoder = self.encoder if encoder is None else encoder
        _decoder = self.decoder if decoder is None else decoder

        if isinstance(text, str):
            token_lines = [_encoder(x + split_on) for x in text.split(split_on)]
        else:
            token_lines = [_encoder(x + split_on) for x in _decoder(text).split(split_on)]
        html = self.generate_style() + self.generate_container(token_lines, strip_padding, pad_token)

        if display_inline:
            display(HTML(html))

        return html

    def __call__(self, text: str, split_on: str = "\n", display_inline: bool = False,
                 encoder: Optional[Callable] = None, decoder: Optional[Callable] = None,
                 strip_padding: bool = True, pad_token: Union[int, str] = 0) -> str:
        """ Makes the TokenVisualization instance callable like a function.

        Args:
            text (str): The text to visualize.
            split_on (str): The delimiter used to split the text into separate lines (default: newline character).
            display_inline (bool): Whether to display the generated HTML inline using IPython display (default: False).
            encoder (Optional[Callable]): An optional encoder function to override the default encoder (default: None).
            decoder (Optional[Callable]): An optional decoder function to override the default decoder (default: None).
            strip_padding (bool, optional): If True, remove padding tokens from the visualization (default: True).
            pad_token (Union[int, str], optional): The value of the padding token (default: 0).

        Returns:
            str: The generated HTML string containing the styled tokens and their container.
        """
        return self.visualize(text, split_on, display_inline, encoder, decoder, strip_padding, pad_token)
