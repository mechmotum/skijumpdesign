---
title: 'skijumpdesign: A Ski Jump Design Tool for Specified Equivalent Fall Height'
tags:
  - python
  - engineering
  - sports
  - ski
  - snowboard
  - safety
authors:
  - name: Jason K. Moore
    orcid: 0000-0002-8698-6143
    affiliation: 1
  - name: Mont Hubbard
    orcid: 0000-0001-8321-6576
    affiliation: 1
affiliations:
 - name: University of California, Davis
   index: 1
date: 24 April 2018
bibliography: paper.bib
---

# Summary
Over the past three decades an evolution has occurred toward freestyle skiing and snowboarding 
involving aerials in terrain parks at ski resorts hosting dedicated
jumping features. Today more than 95% of US ski resorts have such jumps but 
these rarely, if ever, involve formal or detailed design or engineering. 
Although usually tested and modified before being opened to the public, 
they are often simply fabricated based on the past experience of the builder in jump construction. 
Together with the increase in these jumps have come a concomitant increase in injuries and their very high social costs. 
Although omitted here, the voluminous epidemiology and financial effects of these injuries are covered in detail in references 
[@hubbard2009,  @mcneil2012, @levy2015, @petrone2017].

The likelihood and severity of injury on landing are proportional to the energy dissipated on impact, 
when the component of velocity of the jumper perpendicular to the snow surface is brought to zero. This energy is 
naturally measured by the "equivalent fall height" (EFH), defined as the kinetic energy associated with the 
landing velocity component perpendicular to the landing surface divided by mg, where m is the jumper mass and g is the acceleration of gravity.

Past research [@hubbard2009, @swedberg2010, @mcneil2012, @levy2015 ] has developed a theoretical approach for jump design. It is based on 
shaping the landing surface so the perpendicular component of landing velocity (and thus impact landing energy and EFH) 
is controlled to be relatively small everywhere impact is possible. More recent research [@petrone2017] has presented compelling experimental evidence that
these designed jump surfaces embodying low values of EFH are practical to build and, once built, perform as predicted in
limiting landing impact. This experimental research has demonstrated that impact on landing can be controlled through design of the shape of
the landing surface according to the theory.

Ski resorts have been reluctant, however, to adopt this more engineered approach to jump design, in part due to questions
of feasibility, but also becasue of the somewhat ponderous and complex calculations required. Some recent efforts have been made to develop 
numerical software to automate these calculations [@Levy 2015] that also embodies graphical user interfaces but these have relied on proprietary,
closed-source tools and programming environments (MATLAB).  The present open source, online application skijumpdesign is implemented in Python ... etc.   ..., 
removes these restrictions and promises to make the method more widely available to the skiing industry.

Do we need a paragraph here that summarizes in more detail what the software does?

# Acknowledgements

We acknowledge the assistance of Jim McNeil and Dean Levy who aided in testing the application before release.


# References
