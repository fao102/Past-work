Spacecraft Project
Author: Filé Ajanaku
Email: Fajanaku-olaleye1@sheffield.ac.uk

Overview
This project implements a 3D scene of a room on a spaceship using modern OpenGL. The scene includes the following features:

A textured room with walls, a floor, and a ceiling.
A window displaying an animated space view.
Two robots: one dancing robot and one small robot with a spotlight.
A rotating textured globe with a stand.
Interactive controls for lights, animations, and robot movement.
Program Features
Room:

Walls, floor, and ceiling modeled using flat planes.
Back wall features a shiny text texture of my name using diffuse and specular maps (diffuse_[name].jpg, specular_[name].jpg).
Right wall has a repeating texture.
Floor includes a textured path for Robot 2's movement.
Left wall has a large window showing a dynamic space scene.
Robots:

Robot 1 (Dancing Robot):
Built as a hierarchical model with a base, body, legs, arms, and a custom-designed head.
Animates to "dance" with smooth transformations.
Robot 2 (Survey Robot):
Moves along a textured path, changes direction at corners, and features a rotating spotlight attached to its antenna.
Globe:

Textured globe resembling Earth, rotates continuously about its axis.
Globe stand is also textured.
Lighting:

A general world light illuminates the scene.
Robot 2’s spotlight rotates and illuminates parts of the scene (attempted but failed).
Interface controls to toggle or dim both the world light and the spotlight.
User Interface:

Camera controlled with mouse and keyboard (as provided in the tutorial material).
Controls to toggle animations, start/stop robot movement, and adjust lighting.
Animation:

Robot 1's dance animation triggers when Robot 2 is near and can be controlled manually via the interface.
Robot 2 moves along the path, with smooth transitions at corners.
The globe rotates continuously.
The space scene visible through the window changes dynamically over time.

How to Run
Ensure your system is set up with the required Java and JOGL environment.
Extract the contents of the submitted zip file.
Compile the program:
    javac Spacecraft.java
Run the program:
    java Spacecraft
