#pragma once

#include "utils/error_handler.h"

#include <algorithm>
#include <linalg.h>
#include <vector>


using namespace linalg::aliases;

namespace cg
{
	template<typename T>
	class resource
	{
	public:
		explicit resource(size_t size);
		resource(size_t x_size, size_t y_size);
		~resource();

		const T* get_data();
		T& item(size_t item);
		T& item(size_t x, size_t y);

		size_t size_bytes() const;
		size_t count() const;
		size_t get_stride() const;

	private:
		std::vector<T> data;
		size_t item_size = sizeof(T);
		size_t stride;
	};

	template<typename T>
	inline resource<T>::resource(size_t size)
	{
		data.resize(size);
		stride = 0;
	}
	template<typename T>
	inline resource<T>::resource(size_t x_size, size_t y_size)
	{
        data.resize(x_size * y_size);
		stride = x_size;
	}
	template<typename T>
	inline resource<T>::~resource()
	{
	}
	template<typename T>
	inline const T* resource<T>::get_data()
	{
		return data.data();
	}
	template<typename T>
	inline T& resource<T>::item(size_t item)
	{
		return data.at(item);
	}
	template<typename T>
	inline T& resource<T>::item(size_t x, size_t y)
	{
		return data.at(x + y * stride);
	}
	template<typename T>
	inline size_t resource<T>::size_bytes() const
	{
		return data.size() * item_size;
	}
	template<typename T>
	inline size_t resource<T>::count() const
	{
		return data.size();
	}

	template<typename T>
	inline size_t resource<T>::get_stride() const
	{
		return stride;
	}

	struct color
	{
		static color from_float3(const float3& in)
		{
			color out{};
			out.r = in.x;
			out.g = in.y;
			out.b = in.z;
			return out;
		};
		float3 to_float3() const
		{
			return float3(r, g, b);
		}
		float r;
		float g;
		float b;
	};

	struct unsigned_color
	{
		static unsigned_color from_color(const color& color)
		{
			unsigned_color out{};
			out.r = std::clamp(static_cast<int>(color.r * 255), 0, 255);
			out.g = std::clamp(static_cast<int>(color.g * 255), 0, 255);
			out.b = std::clamp(static_cast<int>(color.b * 255), 0, 255);
			return out;
		};
		static unsigned_color from_float3(const float3& color)
		{
            unsigned_color out{};
            out.r = std::clamp(static_cast<int>(color.x * 255), 0, 255);
            out.g = std::clamp(static_cast<int>(color.y * 255), 0, 255);
            out.b = std::clamp(static_cast<int>(color.z * 255), 0, 255);
			return out;
		};
		float3 to_float3() const
		{
			return float3(
					r, g, b
					) / 255;
		};
		uint8_t r;
		uint8_t g;
		uint8_t b;
	};


	struct vertex
	{
		float x, y, z; // coords
		float nx, ny, nz; // normals
		float u, v; // texture coords
		float ar, ag, ab; // ambient
		float dr, dg, db; // diffuse
		float er, eg, eb; // emissive
	};

}// namespace cg