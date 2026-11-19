# Gundam Breaker Importer

Blender 5.2 extension for importing:

- Gundam Breaker PS3 directly from an ISO, extracted folder, or
  `GxArchivedFile*.dat`.
- Gundam Breaker 3 Vita from decrypted game files.

Install this extension ZIP through **Edit > Preferences > Get Extensions >
Install from Disk**. Open **3D View > Sidebar > Gundam Breaker**, choose the
game source, scan, select models, and import.

The extension provides native ISO/archive access, NumPy-accelerated console
model and texture decoding, armatures, skin weights, custom normals,
materials, directly packed texture images, and PS3 MPK clips as Blender 5.2
Actions.

For PS3 animation import, select an imported rig, click **Find Compatible
Animations**, choose motion packages and clips, then click **Import Selected
Animations**.

No game assets, decrypted payloads, or keys are included.

See the project README for full usage and validation details.
