# Gundam Breaker Console Format Notes

These notes document the implementation-level facts used by the importer.
They do not contain game data.

## Resource hashing

Orochi resource names use a 32-bit base-37 hash:

```text
hash = 37
for each UTF-8 byte:
    hash = (hash * 37 + byte) & 0xFFFFFFFF
```

## PS3 game source

The original game's ISO has an ISO9660 directory tree and may store large
members as multiple extents. The importer represents each member as one
segmented random-access stream.

PS3 GDAT tables are big-endian:

```text
"GDAT"
u32 entry_count
repeat entry_count:
    u64 key       # high 32 bits: type hash; low 32 bits: resource hash
    u32 offset
    u32 size
```

PS3 PAC resources use magic `00 63 61 70`, big-endian table fields, 264-byte
entries, and an 8-byte-aligned payload start.

## Vita game source

Vita GDAT tables use little-endian resource hash, type hash, offset, and size
fields. Older split indices can use a matching CSH identity table. Vita PAC
resources use magic `70 61 63 00` and little-endian table fields.

An encrypted NoNpDrm DAT is not a format variant. Its PFS layer must be
decrypted before GDAT parsing.

## BILZ

BILZ fields are big-endian on both supported platforms. Method 0 is stored
data and method 7 is zlib. Some archive resources place the BILZ header at
offset `0x80`.

## Console MDL

Both platforms use compact 80-byte bone records followed by two 64-byte
matrix arrays per bone.

Vita:

- Four-byte prefix and twelve little-endian header fields.
- 112-byte material ranges.
- 56-byte mesh records.
- Eight-byte FVF entries.
- Float32, float16, normalized U8, and U8 index attributes.

PS3:

- Fourteen big-endian header fields.
- 128-byte material ranges.
- 60-byte mesh records.
- RSX float32, float16, normalized byte, integer byte, signed short, and
  packed 11:11:10 attributes.
- The pre-face command area is derived backward from verified section sizes;
  it is not a fixed `buffer_count * record_size` table.
- Physical vertex allocation words occur at `0x70 + buffer_index * 0x7C`.
- Physical buffers are identified when material ranges reset both
  `index_start` and `base_vertex`.

## Compact MDC

- Material names are relative to `0x30`.
- PS3 entries are 36 bytes and big-endian.
- Vita entries are 40 bytes and little-endian.
- Parameter descriptor kind `0x20` is a texture binding.
- Parameter descriptor kind `0x0F` is a four-float value.

## TEX

Vita texture decoding supports BGRA8, A8, BC1, BC2, BC3, and BC7. BC7 is
passed to Blender through an in-memory-generated DDS file and then packed.
All other formats are decoded to NumPy RGBA pixels and uploaded directly.

PS3 compressed blocks are standard linear BC1/BC3 data. Some non-power-of-two
textures store each compressed row at a power-of-two pixel pitch; the importer
decodes that stored width and crops to the logical width.

PS3 format byte `0x46` identifies a BC1 cubemap:

- Six face-major streams.
- Each face contains its mip chain.
- Each face chain is aligned to 256 bytes.
- The importer decodes the six top mips into a 3 x 2 atlas.

## PS3 MPK motion

The original game's motion resource type is `0x3A4305EC`. Character package
paths include an inserted `BigEndian` component:

```text
GP/motion/chr####/BigEndian/chr####
GP/motion/chr####/chr####_gmp###/BigEndian/chr####_gmp###
```

An MPK contains one base section followed by complete animation clips:

```text
u32be section_count
u32be section_offsets[section_count + 1]

section 0: base skeleton
sections 1..section_count-1: clips
```

The base section is:

```text
u8 node_count
u8 reserved[3]

(node_count - 2) records:
    u8 parent_node
    u8 node_type

align 4
u8 joint_ids[node_count - 1]
align 4

node_count transforms:
    f32be translation[3]
    f32be absolute_reference_quaternion_xyzw[4]
```

Node 0 is a global transform. Nodes 1 through `node_count - 1` map in order
to the leading MDL bones. The joint-ID bytes are stored with the root entry
last. Some MDLs contain trailing helper bones that are intentionally absent
from a package. Nodes 0 and 1 have implicit type 0. The remaining node types
are:

| Type | Meaning |
| --- | --- |
| 0 | Ordinary hierarchy node |
| 1 | Two-bone IK chain root |
| 2 | Two-bone IK middle joint |
| 3 | IK endpoint/target |
| 4 | IK endpoint/target with the opposite bend orientation |

Base quaternions are absolute reference orientations, not local hierarchy
rotations. Before evaluating a pose, each non-root reference orientation is
converted with:

```text
local_reference[node] =
    inverse(absolute_reference[parent]) * absolute_reference[node]
```

Each clip begins with:

```text
u16be control_count
u16be frame_count
```

Control types are packed eight at a time in 24-bit big-endian words. The
first control occupies the least-significant three bits:

```text
type[i] = (packed_word >> (3 * (i % 8))) & 7
```

Types 6 and 7 then store an interior-key count followed by interior frame
indices. The count and each index use one byte for clips of at most 255
frames and two big-endian bytes otherwise. Two-byte metadata starts on a
two-byte boundary. Endpoints at frame 0 and `frame_count - 1` are implicit.
The float payload starts on a four-byte boundary.

| Type | Meaning | Payload |
| --- | --- | --- |
| 0 | Constant `0` | None |
| 1 | Rest/base vector sentinel | None |
| 2 | Constant `pi / 2` | None |
| 3 | Constant `pi` | None |
| 4 | Constant `3 pi / 2` | None |
| 5 | Explicit constant | One `f32be` |
| 6 | Smoothstep keys | One value per key |
| 7 | Cubic Hermite keys | Value/tangent pair per key |

For a normalized segment position `t`, type 6 evaluates as:

```text
value = v0 + (v1 - v0) * (3*t*t - 2*t*t*t)
```

Type 7 adds the Hermite tangent contribution:

```text
duration = (right_frame - left_frame) / 30
value += duration * t * (1 - t) * (m0 * (1 - t) - m1 * t)
```

The sampled scalar controls form sparse vectors:

```text
controls 0..1: engine clip controls

global node 0:
    translation
    rotation

for each real node 1..node_count-1:
    translation
    rotation
    scale
```

A vector consumes one type-1 control when it uses the base value; otherwise
it consumes three XYZ controls. The global node has no scale vector.
Ordinary explicit translations are local values for compatible packages.
Explicit rotation vectors are local XYZ Euler radians and are converted
exactly as `Qz * Qy * Qx`. A rest scale is `(1, 1, 1)`.

An explicit translation on a type-3 or type-4 node is different: it is an
absolute endpoint target in source-skeleton coordinates, not that node's
local translation. The importer evaluates the type-1/type-2/type-3-or-4
chain with an analytical two-bone IK solve. It preserves both segment
lengths, uses the sampled pre-IK knee direction as the bend plane, clamps
unreachable targets, reverses the bend side for type 4, and then evaluates
all endpoint descendants from the solved transform.

The importer evaluates every integer source frame at 30 FPS. It converts each
solved absolute transform to a delta from the localized MPK reference pose,
folds node 0 into the first real bone so Blender object placement remains
editable, converts the delta to Blender axes, and writes quaternion Actions
with continuous signs.

## Coordinate conversion

Game coordinates are Y-up. Blender coordinates are Z-up:

```text
Blender X =  Game X
Blender Y = -Game Z
Blender Z =  Game Y
```

The same proper rotation is applied to positions, normals, and bone rest
matrices. UV V coordinates are inverted for Blender image convention.
