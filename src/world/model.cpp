#define TINYOBJLOADER_IMPLEMENTATION

#include "model.h"

#include "utils/error_handler.h"

#include <linalg.h>


using namespace linalg::aliases;
using namespace cg::world;

cg::world::model::model() {}

cg::world::model::~model() {}

void cg::world::model::load_obj(const std::filesystem::path& model_path)
{
	tinyobj::ObjReaderConfig config;
	config.mtl_search_path = model_path.parent_path().string();
	config.triangulate = true;

	tinyobj::ObjReader reader;
	if (!reader.ParseFromFile(model_path.string(), config)) {
		if (!reader.Error().empty()) THROW_ERROR(reader.Error());
	}
	auto& shapes = reader.GetShapes();
	auto& materials = reader.GetMaterials();
	auto& attrs = reader.GetAttrib();

	allocate_buffers(shapes);
	fill_buffers(shapes, attrs, materials, model_path.parent_path());
}

void model::allocate_buffers(const std::vector<tinyobj::shape_t>& shapes)
{
	for (const auto& shape: shapes) {
		size_t inx_offt = 0;
		uint32_t vtx_buf_size = 0;
		uint32_t inx_buf_size = 0;
		std::map<std::tuple<int, int, int>, uint32_t> inx_map;
		const auto& mesh = shape.mesh;

		for (uint8_t face : mesh.num_face_vertices) {
			for (size_t v = 0; v < face; v++) {
				tinyobj::index_t inx = mesh.indices[inx_offt + v];
				auto inx_tuple = std::make_tuple(
						inx.vertex_index,
						inx.normal_index,
						inx.texcoord_index);
				if (inx_map.count(inx_tuple) == 0) {
					inx_map[inx_tuple] = vtx_buf_size;
					vtx_buf_size++;
				}
				inx_buf_size++;
			}
			inx_offt += face;
		}
		vertex_buffers.push_back(std::make_shared<cg::resource<cg::vertex>>(vtx_buf_size));
		index_buffers.push_back(std::make_shared<cg::resource<uint32_t>>(inx_buf_size));
	}
	textures.resize(shapes.size());
}

float3 cg::world::model::compute_normal(const tinyobj::attrib_t& attrib, const tinyobj::mesh_t& mesh, size_t index_offset)
{
	auto a_id = mesh.indices[index_offset];
	auto b_id = mesh.indices[index_offset + 1];
	auto c_id = mesh.indices[index_offset + 2];

	float3 a(
			attrib.vertices[3 * a_id.vertex_index],
			attrib.vertices[3 * a_id.vertex_index + 1],
			attrib.vertices[3 * a_id.vertex_index + 2]);
	float3 b(
			attrib.vertices[3 * b_id.vertex_index],
			attrib.vertices[3 * b_id.vertex_index + 1],
			attrib.vertices[3 * b_id.vertex_index + 2]);
	float3 c(
			attrib.vertices[3 * c_id.vertex_index],
			attrib.vertices[3 * c_id.vertex_index + 1],
			attrib.vertices[3 * c_id.vertex_index + 2]);
	return normalize(cross(b - a, c - a));
}

void model::fill_vertex_data(cg::vertex& vertex, const tinyobj::attrib_t& attrib, const tinyobj::index_t idx, const float3 computed_normal, const tinyobj::material_t material)
{
    vertex.x = attrib.vertices[3 * idx.vertex_index];
    vertex.y = attrib.vertices[3 * idx.vertex_index + 1];
    vertex.z = attrib.vertices[3 * idx.vertex_index + 2];

	if (idx.normal_index < 0) {
		vertex.nx = computed_normal.x;
		vertex.ny = computed_normal.y;
		vertex.nz = computed_normal.z;
	} else {
		vertex.nx = attrib.normals[3 * idx.normal_index];
		vertex.ny = attrib.normals[3 * idx.normal_index + 1];
		vertex.nz = attrib.normals[3 * idx.normal_index + 2];
	}

	if (idx.texcoord_index < 0) {
		vertex.u = 0;
		vertex.v = 0;
	} else {
		vertex.u = attrib.texcoords[2 * idx.texcoord_index];
		vertex.v = attrib.texcoords[2 * idx.texcoord_index + 1];
	}

	vertex.ar = material.ambient[0];
	vertex.ag = material.ambient[1];
	vertex.ab = material.ambient[2];

    vertex.dr = material.diffuse[0];
    vertex.dg = material.diffuse[1];
    vertex.db = material.diffuse[2];

    vertex.er = material.emission[0];
    vertex.eg = material.emission[1];
    vertex.eb = material.emission[2];


}

void model::fill_buffers(const std::vector<tinyobj::shape_t>& shapes, const tinyobj::attrib_t& attrib, const std::vector<tinyobj::material_t>& materials, const std::filesystem::path& base_folder)
{
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t inx_offt = 0;
        uint32_t vtx_buf_id = 0;
        uint32_t inx_buf_id = 0;
		auto vtx_buf = vertex_buffers[s];
		auto inx_buf = index_buffers[s];
        std::map<std::tuple<int, int, int>, uint32_t> inx_map;
        const auto& mesh = shapes[s].mesh;

        for (size_t f = 0; f < mesh.num_face_vertices.size(); f++) {
			uint8_t face = mesh.num_face_vertices[f];
			float3 normal;
			if (mesh.indices[inx_offt].normal_index < 0) {
				normal = compute_normal(attrib, mesh, inx_offt);
			}
            for (size_t v = 0; v < face; v++) {
                tinyobj::index_t inx = mesh.indices[inx_offt + v];
                auto inx_tuple = std::make_tuple(
                        inx.vertex_index,
                        inx.normal_index,
                        inx.texcoord_index);
                if (inx_map.count(inx_tuple) == 0) {
                    cg::vertex& vtx = vtx_buf->item(vtx_buf_id);
					const auto& mat = materials[mesh.material_ids[f]];
					fill_vertex_data(vtx, attrib, inx, normal, mat);

					inx_map[inx_tuple] = vtx_buf_id++;
                }
				inx_buf->item(inx_buf_id) = inx_map[inx_tuple];
                inx_buf_id++;
            }
            inx_offt += face;
        }
		if (!materials[mesh.material_ids[0]].diffuse_texname.empty()) {
			textures[s] = base_folder / materials[mesh.material_ids[0]].diffuse_texname;
		}
    }
    textures.resize(shapes.size());
}


const std::vector<std::shared_ptr<cg::resource<cg::vertex>>>&
cg::world::model::get_vertex_buffers() const
{
	return vertex_buffers;
}

const std::vector<std::shared_ptr<cg::resource<unsigned int>>>&
cg::world::model::get_index_buffers() const
{
	return index_buffers;
}

const std::vector<std::filesystem::path>& cg::world::model::get_per_shape_texture_files() const
{
	return textures;
}


const float4x4 cg::world::model::get_world_matrix() const
{
	return float4x4{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1}};
}
