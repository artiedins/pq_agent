Create a static 3D pergola scene using three.js.

All dimensions are in feet. Y-axis is up.

**Scene Setup:**
1.  **Ground:** grassy plain
2.  **Lighting:** ambient light and directional light
3.  **Material:** wood for all pergola parts
4.  **Camera:** Position for a good overview

**Pergola Dimensions & Components (Actual lumber dimensions in feet):**
*   **Post dimensions (nominal 4x4):** width/depth = 3.5/12, height = 8.0.
*   **Main Beam dimensions (nominal 2x8):** thickness (Z-axis for beam along X) = 1.5/12, height (Y-axis) = 7.25/12. Overhang = 1.5.
*   **Rafter dimensions (nominal 2x4):** thickness (X-axis for rafter along Z) = 1.5/12, height (Y-axis) = 3.5/12. Overhang = 1.5.
*   **Purlin dimensions (nominal 2x2):** width/height/depth = 1.5/12.

**Construction:**

1.  **Posts (4 count):**
    *   Calculate position so as to create a 9ft (X) by 6ft (Z) rectangle.

2.  **Main Beams (2 count):**
    *   Calculate position so as to ride on top of posts
    *   Total Length: 9.0 (post span) + 2 * 1.5 (overhangs) = 12.0.
    *   Height: 7.25/12. Thickness: 1.5/12. Overhang: 1.5.
    *   Cut a tapered relief on the bottom face of each 1.5 overhang, reducing the lumber height to 35% of original by the end of each overhang to create an upward sweep profile, keeping the top face flat.

3.  **Rafters (7 count):**
    *   Calculate positions so as to ride on top of main beams, with end rafters aligning with the ends of the main beam ends.
    *   Spacing: Distribute 7 rafters evenly along the 12.0 ft X-length of main beams
    *   Total Length: 6.0 (main beam span based on post Z-centers) + 2 * 1.5 (overhangs) = 9.0.
    *   Height: 3.5/12. Thickness: 1.5/12. Overhang: 1.5.
    *   Cut a tapered relief on the bottom face of each 1.5 overhang, reducing the lumber height to 35% of original by the end of each overhang to create an upward sweep profile, keeping the top face flat.

4.  **Purlins (7 count):**
    *   Calculate positions so as to ride on top of rafters, with end purlins aligning with the ends of the rafter ends.
    *   Spacing: Distribute 7 purlins evenly along the 9.0 ft Z-length of rafters.
    *   Dimensions: Length = 12.0. Height = 1.5/12. Depth = 1.5/12.

Add OrbitControls for ease of viewing. You may add minor embellishments but do not add anything that might distract from the main pergola design.
Think very hard as this is a tricky spatial reasoning challenge posed as a 3D modeling problem.
Output the pergola you designed as a three.js visualization in a single html file.
