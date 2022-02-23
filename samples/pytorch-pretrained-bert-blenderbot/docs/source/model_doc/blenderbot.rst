Blenderbot
----------------------------------------------------
**DISCLAIMER:** If you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`

Overview
~~~~~~~~~~~~~~~~~~~~~

The Blender chatbot model was `proposed in Recipes for building an open-domain chatbot <https://arxiv.org/pdf/2004.13637.pdf>`_ Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston on 30 Apr 2020.
Here the abstract,

Building open-domain chatbots is a challenging area for machine learning research. While prior work has shown that scaling neural models in the number of parameters and the size of the data they are trained on gives improved results, we show that other ingredients are important for a high-performing chatbot. Good conversation requires a number of skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to their partners, and displaying knowledge, empathy and personality appropriately, while maintaining a consistent persona. We show that large scale models can learn these skills when given appropriate training data and choice of generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter models, and make our models and code publicly available. Human evaluations show our best models are superior to existing approaches in multi-turn dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing failure cases of our models.

The The Authors' code can be found `here <https://github.com/facebookresearch/ParlAI>`_


Implementation Notes
~~~~~~~~~~~~~~~~~~~~

Blenderbot uses a standard seq2seq model transformer <https://arxiv.org/pdf/1706.03762.pdf> based architecture


BlenderbotConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BlenderbotConfig
    :members:


BlenderbotTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BlenderbotTokenizer
    :members: build_inputs_with_special_tokens


BlenderbotSmallTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BlenderbotSmallTokenizer
    :members: bpe, convert_tokens_to_string, save_vocabulary


BlenderbotForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BlenderbotForConditionalGeneration
    :members: generate, forward
