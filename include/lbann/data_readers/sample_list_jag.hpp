#ifndef __SAMPLE_LIST_JAG_HPP__
#define __SAMPLE_LIST_JAG_HPP__

#include <iostream>
#include <string>
#include <vector>
#include <functional>

#ifndef _JAG_OFFLINE_TOOL_MODE_
#include "lbann/comm.hpp"
#else
#include <mpi.h>
#endif

#include "lbann/utils/file_utils.hpp"
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/utility.hpp>
#include "conduit/conduit_relay_io_hdf5.hpp"

#define LBANN_MAX_OPEN_DATA_FILES 384

namespace lbann {

struct sample_list_header {
  bool m_is_exclusive;
  /// Number of included samples
  size_t m_included_sample_count;
  /// Number of excluded samples
  size_t m_excluded_sample_count;
  size_t m_num_files;
  std::string m_file_dir;
  std::string m_sample_list_filename;

  sample_list_header();

  bool is_exclusive() const;
  size_t get_sample_count() const;
  size_t get_num_files() const;
  const std::string& get_sample_list_filename() const;
  const std::string& get_file_dir() const;
  template <class Archive> void serialize( Archive & ar ) {
    ar(m_is_exclusive, m_included_sample_count, m_excluded_sample_count, m_num_files, m_file_dir, m_sample_list_filename);
  }
};

/**
 * Maps a global index of a sample list to a local index.
 * When managing the sample list in a distributed fashion, with which every
 * one has the same copy (the whole global list), m_partition_offset must be
 * zero. In this case, the local index is the same as the global index.
 * When managing the sample list in a centralized fashion, with which each
 * has a portion of the list that corresponds to the only samples it needs,
 * a global index is subtracted by m_partition_offset for local indexing.
 */
struct sample_list_indexer {
  sample_list_indexer();
  size_t operator()(size_t idx) const;

  void set_partition_offset(size_t o);
  size_t get_partition_offset() const;
  bool check_index(size_t i) const;

  size_t m_partition_offset;
};

static const std::string conduit_hdf5_exclusion_list = "CONDUIT_HDF5_EXCLUSION";
static const std::string conduit_hdf5_inclusion_list = "CONDUIT_HDF5_INCLUSION";

class sample_list_jag {
 public:
  /// The type of the native identifier of a sample rather than an arbitrarily assigned index
  using sample_name_t = std::string;
  /// The type for arbitrarily assigned index
  using sample_id_t = std::size_t;
  /// To describe a sample as a pair of the file to which it belongs and its name
  //  using sample_t = std::pair<std::string, sample_name_t>;
  using sample_t = std::pair<sample_id_t, sample_name_t>;
  /// Map of the file index to the file name, file descriptor, and
  /// and a queue of each step and substep when data will be loaded from the file
  using sample_id_map_t = std::tuple<std::string, hid_t, std::deque<std::pair<int,int>>>; // rename
                                                                            // to sample_to_file_map
  /// Type for the list of samples
  using samples_t = std::vector< sample_t >;
  using samples_id_map_v_t = std::vector< sample_id_map_t >; // rename to sample_to_file_v or something
  /// Type for the map of file descriptors to usage step and substep
  using fd_use_map_t = std::pair<sample_id_t, std::pair<int,int>>;

  sample_list_jag();
  ~sample_list_jag();

  /// Set the number of partitions and clear internal states
  void set_num_partitions(size_t n);

  /// Set the index mapping function
  void set_indexer(const sample_list_indexer& indexer);

  /// Get the index mapping function
  const sample_list_indexer& get_indexer() const;

  /// Load a sample list file
  void load(const std::string& samplelist_file, size_t stride=1, size_t offset=0);

  /// Load the header of a sample list file
  sample_list_header load_header(const std::string& samplelist_file) const;

  /// Extract a sample list from a serialized sample list in a string
  void load_from_string(const std::string& samplelist);

  /// Tells how many samples in the list
  size_t size() const;

  /// Tells if the internal list is empty
  bool empty() const;

  /// Clear internal states
  void clear();

  template <class Archive> void serialize( Archive & ar );

  /// Check if a sample index is in the valid range
  bool check_index(size_t idx) const;

  /// Serialize sample list for a partition
  bool to_string(size_t p, std::string& sstr) const;

  /// Serialize sample list for all partitions
  bool to_string(std::string& sstr) const;

  /// Write the sample list of partition p
  void write(size_t p, const std::string filename) const;

  /// Write the sample list of each partitions
  void write(const std::string filename) const;

  /// Allow read-only access to the internal list data
  const samples_t& get_list() const;

  /// Copy the internal list data for partition p
  bool get_list(size_t p, samples_t& l_p) const;

  /// Allow read-only access to the internal list data for partition p via iterators
  std::pair<samples_t::const_iterator, samples_t::const_iterator> get_list(size_t p) const;

  /// Allow the read-only access to the list header
  const sample_list_header& get_header() const;

  /// Allow read-only access to the metadata of the idx-th sample in the list
  const sample_t& operator[](size_t idx) const;

  const std::string& get_samples_filename(sample_id_t id) const {
    return std::get<0>(m_sample_id_map[id]);
  }

  const std::string& get_samples_dirname() const {
    return m_header.get_file_dir();
  }

  hid_t get_samples_hdf5_handle(sample_id_t id) const {
    hid_t h = std::get<1>(m_sample_id_map[id]);
    return h;
  }

  void set_samples_filename(sample_id_t id, const std::string& filename) {
    std::get<0>(m_sample_id_map[id]) = filename;
  }

  void set_samples_hdf5_handle(sample_id_t id, hid_t h) {
#if 0
    int bucket_count = m_open_fd_map.bucket_count();
    int bucket = m_open_fd_map.bucket(filename);
    if(!allow_collisions && m_open_fd_map.bucket_size(bucket) > 0) {
      if(m_open_fd_map.bucket_size(bucket) != 1) {
        LBANN_ERROR(std::string{} + " :: unexpected number of open file descriptors for bucket "
                    + std::to_string(bucket));
      }
      auto local_it = m_open_fd_map.begin(bucket);
      if(local_it == m_open_fd_map.end(bucket)) {
        LBANN_ERROR(std::string{} + " :: bucket '" + std::to_string(bucket)
                    + "' has an empty iterator");
      }
      const std::string& old_filename = local_it->first;
      hid_t old_h = local_it->second;
      if (old_h <= static_cast<hid_t>(0)) {
        LBANN_ERROR(std::string{} + " :: data file '" + old_filename
                    + "' has a corrupt file descriptor = " + std::to_string(old_h));
      }

      conduit::relay::io::hdf5_close_file(old_h);
      int num_erased = m_open_fd_map.erase(old_filename);
      if(num_erased != 1) {
        LBANN_ERROR(std::string{} + " :: erasing file descriptor for '" + old_filename
                    + "' that had a file descriptor = " + std::to_string(old_h));
      }
    }

    if(m_open_fd_pq.size() > 100/*LBANN_MAX_OPEN_DATA_FILES*/) {
      std::cout << "The file descriptors are over the limit, lets close " << m_open_fd_pq.top().first << std::endl;
      while(!m_open_fd_pq.empty()) {
        auto e = m_open_fd_pq.top();
        std::cout << "{" << e.first << ", " << e.second << "}" << std::endl;
        //        std::cout << q.top() << " ";
        m_open_fd_pq.pop();
      }
      std::cout << '\n';
    }

    auto result = m_open_fd_map.emplace(filename, h);
    m_open_fd_pq.emplace(std::make_pair(filename,access_count));

    int bucket2 = m_open_fd_map.bucket(filename);
    int bucket_count2 = m_open_fd_map.bucket_count();
    if(!result.second) {
      LBANN_WARNING(std::string{} + " :: The key for " + filename + " already existed");
    }
    if(bucket2 != bucket) {
        LBANN_ERROR(std::string{} + " :: the buckets don't match original bucket "
                    + std::to_string(bucket) + " with a count of " + std::to_string(bucket_count) + " and new bucket " + std::to_string(bucket2) + " and a new count of " + std::to_string(bucket_count2));
    }
    if(m_open_fd_map.bucket_size(bucket) != 1) {
        LBANN_WARNING(std::string{} + " :: there should be one entry with an open file descriptors for bucket "
                      + std::to_string(bucket) + " not "
                      + std::to_string(m_open_fd_map.bucket_size(bucket)) + " entries");
    }
#endif

    //    for (auto&& e : m_sample_id_map) {
    auto&& e = m_sample_id_map[id];
    std::get<1>(e) = h;
    // std::cout << "Attempt to set the hdf5 handle " << h << " for filename " << std::get<0>(e) << std::endl;

    // std::cout << "set_files_hdf5_handle existing list for " << id << " {" << std::get<0>(e) << ", " << std::get<1>(e) << ": ";
    // for (auto&& v : std::get<2>(e)) {
    //   std::cout << "{" << v.first << "," << v.second << "}, ";
    // }
    // std::cout << std::endl;

    // if(!m_open_fd_pq.empty()) {
    //   // std::cout << "set_files_hdf5_handle Priotirty QUeue ";
    //   std::make_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
    //   // auto& q = m_open_fd_pq.front();
    //   // std::cout << q.first << " {" << q.second.first << "," << q.second.second << "}, ";
    //   // std::cout << std::endl;
    // }

    if(!m_open_fd_pq.empty()) {
      /// Before we can enqueue the any new access times for this descriptor, remove any
      /// earlier descriptor
      std::sort_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
      if(m_open_fd_pq.front().first == id) {
        //LBANN_ERROR("We have weirdness here, the head of the queue is not " + std::to_string(id));
        m_open_fd_pq.pop_front();
      }
      std::make_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
    }

    auto& file_access_queue = std::get<2>(e);
    if(!file_access_queue.empty()) {
      file_access_queue.pop_front();
      if(!file_access_queue.empty()) {
        m_open_fd_pq.emplace_back(std::make_pair(id,file_access_queue.front()));
        std::push_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
        // std::cout << "set_files_hdf5_handle New priotirty queue top ";
        //        auto& q = m_open_fd_pq.front();
        // for(auto&& q: m_open_fd_pq) {
        //   std::cout << q.first << " {" << q.second.first << "," << q.second.second << "}, ";
        // }
        // std::cout << std::endl;
      }
      // std::cout << "set_files_hdf5_handle updated list for " << id << " {" << std::get<0>(e) << ", " << std::get<1>(e) << ": ";
      // for (auto&& v : std::get<2>(e)) {
      //   std::cout << "{" << v.first << "," << v.second << "}, ";
      // }
      // std::cout << std::endl;
    }

    //        std::get<1>(m_sample_id_map[id]) = h;
    //        std::cout << "I am setting the hdf5 handle " << h << " for filename " << filename << std::endl;

    //    m_open_fd_map.emplace(std::make_tuple(filename, h, access_count));
    // for (auto&& e : m_sample_id_map) {
    //   std::cout << "set_files_hdf5_handle {" << std::get<0>(e) << ", " << std::get<1>(e) << ": ";
    //   if(std::get<2>(e).empty()) {
    //     std::cout << "empty" << std::endl;
    //   }else {
    //     for (auto&& v : std::get<2>(e)) {
    //       std::cout << "{" << v.first << "," << v.second << "}, ";
    //     }
    //     std::cout << std::endl;
    //   }
    // }
    // for (auto&& e : m_sample_id_map)
    //   std::cout << "{" << std::get<0)>(e) << ", " << std::get<1>(e) << ", " << std::get<2>(e) << "}" << std::endl;

  }

  void set_files_hdf5_handle(const std::string& filename, hid_t h) {
    sample_id_t id = 0;
    for (auto&& e : m_sample_id_map) {
      if(std::get<0>(e) == filename) {
        break;
      }
      id++;
    }
    set_samples_hdf5_handle(id, h);
  }

  hid_t open_samples_hdf5_handle(const size_t i) {
    const sample_t& s = m_sample_list[i];
    sample_id_t id = s.first;
    hid_t h = get_samples_hdf5_handle(id);
    if (h <= static_cast<hid_t>(0)) {
      const std::string& file_name = get_samples_filename(id);
      const std::string conduit_file_path = add_delimiter(get_samples_dirname()) + file_name;
      if (file_name.empty() || !check_if_file_exists(conduit_file_path)) {
        LBANN_ERROR(std::string{} + " :: data file '" + conduit_file_path + "' does not exist.");
      }
      h = conduit::relay::io::hdf5_open_file_for_read( conduit_file_path );
      if (h <= static_cast<hid_t>(0)) {
        LBANN_ERROR(std::string{} + " :: data file '" + conduit_file_path + "' could not be opened.");
      }
      set_samples_hdf5_handle(id, h);
    }else {

      if(!m_open_fd_pq.empty()) {
        /// Before we can enqueue the any new access times for this descriptor, remove any
        /// earlier descriptor
        std::sort_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
        if(m_open_fd_pq.front().first == id) {
          //          LBANN_ERROR("We have weirdness here, the head of the queue is not " + std::to_string(id));
          m_open_fd_pq.pop_front();
        }
        std::make_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
      }

      // std::sort_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
      // if(!m_open_fd_pq.empty() && m_open_fd_pq.front().first != id) {
      //   LBANN_WARNING("We have weirdness here, the head of the queue is not " + std::to_string(id));
      // }

      auto& e = m_sample_id_map[id];

      // std::cout << "open_files_hdf5_handle updated list {" << std::get<0>(e) << ", " << std::get<1>(e) << ": ";
      // for (auto&& v : std::get<2>(e)) {
      //   std::cout << "{" << v.first << "," << v.second << "}, ";
      // }
      // std::cout << std::endl;

      //        std::make_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
      // if(!m_open_fd_pq.empty()) {
      //   // std::cout << "open_files_hdf5_handle priority queue :";
      //   auto& p = m_open_fd_pq.front();
      //   std::cout << "[" << p.first << ", " << "{" << p.second.first << "," << p.second.second << "}], ";
      //   std::cout << std::endl;
      // }

      auto& file_access_queue = std::get<2>(e);
      if(!file_access_queue.empty()) {
        file_access_queue.pop_front();
        if(!file_access_queue.empty()) {
          m_open_fd_pq.emplace_back(std::make_pair(id,file_access_queue.front()));
          std::push_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
          // if(!m_open_fd_pq.empty()) {
          //   std::cout << "open_files_hdf5_handle new priority queue :";
          //   //            auto& p = m_open_fd_pq.front();
          //   for(auto&& p: m_open_fd_pq) {
          //     std::cout << "[" << p.first << ", " << "{" << p.second.first << "," << p.second.second << "}], ";
          //   }
          //   std::cout << std::endl;
          // }
        }
      }
    }

    return h;
  }

  void all_gather_archive(const std::string &archive, std::vector<std::string>& gathered_archive, lbann_comm& comm);
  template<typename T> size_t all_gather_field(T data, std::vector<T>& gathered_data, lbann_comm& comm);
  void all_gather_packed_lists(lbann_comm& comm);

  void compute_epochs_file_usage(const std::vector<int>& shufled_indices, int mini_batch_size, const lbann_comm& comm);

 protected:

  /// Reads a header line from the sample list given as a stream, and use the info string for error message
  std::string read_header_line(std::istream& ifs, const std::string& filename, const std::string& info) const;

  /// Reads the header of a sample list
  sample_list_header read_header(std::istream& istrm, const std::string& filename) const;

  /// Get the list of samples that exist in a conduit bundle
  hid_t get_conduit_bundle_samples(std::string conduit_file_path, std::vector<std::string>& sample_names, size_t included_samples, size_t excluded_samples);

  /// read the body of exclusive sample list
  void read_exclusive_list(std::istream& istrm, size_t stride=1, size_t offset=0);

  /// read the body of inclusive sample list
  void read_inclusive_list(std::istream& istrm, size_t stride=1, size_t offset=0);

  /// Reads a sample list and populates the internal list
  size_t get_samples_per_file(std::istream& istrm, const std::string& filename, size_t stride=1, size_t offset=0);

  /// Compute the sample index range that partition p covers
  void get_sample_range_per_part(const size_t p, size_t& sid_start, size_t& sid_end) const;

  /// Add the header info to the given string
  void write_header(std::string& sstr, size_t num_files) const;

  static bool pq_cmp(fd_use_map_t left, fd_use_map_t right) {
    return ((left.second).first < (right.second).first) ||
           (((left.second).first == (right.second).first) &&
            ((left.second).second < (right.second).second)); }

 private:

  /// The number of partitions to divide samples into
  size_t m_num_partitions;

  /// header info of sample list
  sample_list_header m_header;

  /// Contains list of all sample
  samples_t m_sample_list;

  /// Maps sample IDs to file names, file descriptors, and use counts
  samples_id_map_v_t m_sample_id_map;

  /// Maps a global index to a local index
  sample_list_indexer m_indexer;

  /// Track the number of samples per file
  std::unordered_map<std::string, size_t> m_file_map;

  /// Track the number of open file descriptors and how many times
  /// each file descriptor will be used
  //  std::unordered_map<std::string, hid_t> m_open_fd_map;
  //  std::set<fd_use_map, std::function<bool(fd_use_map_t, fd_use_map_t)>> m_open_fd_map;
   // Using lambda to compare elements.
  //  auto cmp = [](std::pair<string, int> left,std::pair<string, int> right) { return (left.second) > (right.second);};
  //  std::priority_queue<fd_use_map_t, std::vector<fd_use_map_t>, std::function<bool(fd_use_map_t, fd_use_map_t)>> m_open_fd_pq;
  std::deque<fd_use_map_t> m_open_fd_pq;

};

void handle_mpi_error(int ierr);

#ifndef _JAG_OFFLINE_TOOL_MODE_
void distribute_sample_list(const sample_list_jag& sn,
                            std::string& my_samples,
                            lbann_comm& comm);
#else
void distribute_sample_list(const sample_list_jag& sn,
                            std::string& my_samples,
                            MPI_Comm& comm);
#endif

} // end of namespace

#include "sample_list_jag_impl.hpp"

#endif // __SAMPLE_LIST_JAG_HPP__
