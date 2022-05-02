use bytemuck::{Pod, Zeroable};
use gltf::{Accessor, Gltf, Semantic};
use std::{fs, mem, path::Path};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position);

#[derive(Debug, Default)]
pub struct Mesh {
    pub vertex_bytes: Vec<Vertex>,
    pub index_bytes: Vec<u16>,
}

#[derive(Debug, Default)]
pub struct Model {
    pub meshes: Vec<Mesh>,
}

pub fn load<P>(path: P) -> Model
where
    P: AsRef<Path> + Clone,
{
    let gltf = Gltf::open(path.clone()).unwrap();

    let mut model = Model::default();

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let positions_accessor = primitive
                .attributes()
                .find_map(|(semantic, accesor)| {
                    if semantic == Semantic::Positions {
                        Some(accesor)
                    } else {
                        None
                    }
                })
                .unwrap();

            let index_byte_size = primitive.indices().unwrap().size();
            if index_byte_size != 2 {
                panic!(
                    "primitive index byte size is not 4 bytes: {}",
                    index_byte_size
                );
            }

            let positions = get_data_from_accessor::<Vertex, P>(path.clone(), &positions_accessor);
            let indices =
                get_data_from_accessor::<u16, P>(path.clone(), &primitive.indices().unwrap());

            model.meshes.push(Mesh {
                vertex_bytes: positions,
                index_bytes: indices,
            });
        }
    }

    model
}

fn get_data_from_accessor<T, P>(path: P, accessor: &Accessor) -> Vec<T>
where
    P: AsRef<Path>,
{
    let view = accessor.view().unwrap();
    let buffer = view.buffer();

    let buffer_data = match buffer.source() {
        gltf::buffer::Source::Bin => todo!(),
        gltf::buffer::Source::Uri(uri) => {
            let buffer_path = path.as_ref().parent().unwrap().join(uri);
            load_buffer_data_from_uri(buffer_path)
        }
    };

    let from = view.offset() + accessor.offset();
    let to = view.offset() + accessor.offset() + accessor.count() * accessor.size();

    let view_data = &buffer_data[from..to];

    let data = match view.stride() {
        Some(stride) => {
            view_data
                .iter()
                .enumerate()
                .filter_map(|(index, byte)| {
                    if index % stride < accessor.size() {
                        Some(*byte)
                    } else {
                        None
                    }
                })
                .collect::<Vec<u8>>()
        }
        None => view_data.to_vec(),
    };

   transmute_byte_vec::<T>(data)
}

fn load_buffer_data_from_uri<P>(path: P) -> Vec<u8>
where
    P: AsRef<Path>,
{
    fs::read(path).unwrap()
}

fn transmute_byte_vec<T>(mut bytes: Vec<u8>) -> Vec<T> {
    unsafe {
        let size_of_t = mem::size_of::<T>();
        let length = bytes.len() / size_of_t;
        let capacity = bytes.capacity() / size_of_t;
        let mutptr = bytes.as_mut_ptr() as *mut T;
        mem::forget(bytes);

        Vec::from_raw_parts(mutptr, length, capacity)
    }
}
