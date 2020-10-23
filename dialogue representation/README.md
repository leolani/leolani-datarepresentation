
## The pipeline
A pipeline is composed of several modules or channels.



### Channels / Modules
Channels may also be thought of as modules that processed some data and transformed or extracted information. 

* ``` "source": null ``` means the output of this module is not available in the dataset, but the input is available to be processed by the default modules
* ``` "input": null ``` means the output of this module is actually just a raw multimodal input, instead of the result of a processing module or an annotation


* ```"Text to triple"``` could be replaced with NERC and Relation Extraction



## States 
Components change according to events. They have a state that we an record

* VAD (maybe?)
* Context
* Brain 