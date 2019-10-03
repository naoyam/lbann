// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// // may not use this file except in compliance with the License.  You may
// // obtain a copy of the License at:
// //
// // http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// // implied. See the License for the specific language governing
// // permissions and limitations under the license.
// //
// ////////////////////////////////////////////////////////////////////////////////
//
//
#ifndef LBANN_DATA_READER_HDF5_HPP
#define LBANN_DATA_READER_HDF5_HPP
#include "data_reader_image.hpp"
#include "hdf5.h"

namespace lbann {
    /**
     * Data reader for data stored in hdf5 files will need to assume the file contains x
     */
    class hdf5_reader : public image_data_reader {
    public:
        hdf5_reader(const bool shuffle);
        hdf5_reader* copy() const override { return new hdf5_reader(*this); }

        std::string get_type() const override {
            return "data_reader_hdf5_images"; 
        }
        //void set_input_params(int width, int height, int depth, int num_ch, int num_labels);
        void load() override;
        /// Set whether to fetch labels
        void set_has_labels(bool b)  {m_has_labels = b; }
        void set_has_responses(bool b) { m_has_responses = b; }
        void set_linearized_image_size();
        void set_defaults() override;

    protected:
        void read_hdf5(hsize_t h_data, hsize_t filespace, int rank, std::string key, hsize_t* dims);
        //void set_defaults() override;
        bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
        bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;
        bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;
        /// Whether to fetch a label from the last column.
        bool m_has_labels = false;
        /// Whether to fetch a response from the last column.
        bool m_has_responses = false;
        int m_image_depth=0; 
        std::vector<short int*> m_image_data;
    private:
        static const std::string HDF5_KEY_DATA, HDF5_KEY_LABELS, HDF5_KEY_RESPONSES;
    };
}
#endif
