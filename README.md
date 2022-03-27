# Orbifold Visualization

## Task specification

Create a ray tracing program that displays a dodecahedron room that can be written into a sphere with a radius of √3 m. The room has an optically smooth gold object defined by the implicit equation f(x, y, z) = exp⁡ (ax ^ 2 + by ^ 2-cz) -1, cut into a sphere of radius 0.3 m in the center of the room, and has a point light source. The walls of the room are of the diffuse-speculative type from the corner to 0.1 m, within which are portals opening to another similar rooms, rotated 72 degrees around the center of the wall and mirrored to the plane of the wall. The light source is not lit through the portal, each room has its own light source. It is enough to cross the portals a maximum of 5 times during the display. The virtual camera looks at the center of the room and revolves around it.

Refractive index and extinction coefficient of gold: n / k: 0.17 / 3.1, 0.35 / 2.7, 1.5 / 1.9

The other parameters can be selected individually so that the image is beautiful. A, b, c are positive, non-integer numbers.

## Solution

<img width="599" alt="Screenshot 2022-03-27 at 22 03 10" src="https://user-images.githubusercontent.com/27449756/160298835-c37a4261-7602-40ad-921d-4875c2d833dd.png">
