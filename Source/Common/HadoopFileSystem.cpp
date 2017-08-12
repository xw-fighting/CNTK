//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include <string>
#include <stdlib.h>

#include "File.h"
#include "HadoopFileSystem.h"

#ifdef USE_HDFS
#include <linux/limits.h>
#endif

#define HDFS_ERROR -1

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

hdfs_File::hdfs_File(const wstring filename, int fileOptions)
{
#ifdef USE_HDFS
    int loc = filename.find(":");
    string nn = filename.substr(0, loc);
    filename = filename.substr(1);
    loc = filename.find("/");
    int port = stoi(filename.substr(0, loc));
    m_filename = filename.substr(loc + 1);

    m_fs = hdfsConnect(nn, port);
    if (m_fs == nullptr)
    {
        RuntimeError("File: failed to connect to HDFS.")
    }

    const auto reading = !!(fileOptions & fileOptionsRead);
    const auto writing = !!(fileOptions & fileOptionsWrite);
    if (!reading && !writing)
        RuntimeError("File: either fileOptionsRead or fileOptionsWrite must be specified");
    m_seekable = reading ? true : false;
    m_options  = reading ? O_RDONLY : O_WRONLY;
    m_file = hdfsOpenFile(m_fs, m_filename, m_options, 0, 0, 0);
    if (m_file == nullptr)
    {
        RuntimeError("File: failed to open the file on HDFS.")
    }
#else
    RuntimeError("File: CNTK was built without HDFS support.");
#endif
}

hdfs_File::~hdfs_File()
{
#ifdef USE_HDFS
    if (hdfsDiconnect(m_fs) == HDFS_ERROR)
        RuntimeError("File: failed to disconnect with HDFS.");
#else
    RuntimeError("File: CNTK was built without HDFS support.");
#endif
}

void hdfs_File::Flush()
{
#ifdef USE_HDFS
    if (hdfsFlush(m_fs, m_file) == HDFS_ERROR)
    {
        RuntimeError("File: failed to flush file to HDFS.");
    }
#else
    RuntimeError("File: CNTK was built without HDFS support.");
#endif
}

size_t hdfs_File::Size()
{
#ifdef USE_HDFS
    hdfs_FileInfo* fileInfo = hdfsGetPathInfo(m_fs, m_filename);
    if (fileInfo == nullptr)
    {
        RuntimeError("File: failed to get File Info.");
    }
    return (size_t)fileInfo->mSize;
#else
    RuntimeError("File: CNTK was built without HDFS support.");
#endif
}

uint64_t hdfs_File::GetPosition()
{
#ifdef USE_HDFS
    if (!CanSeek())
        RuntimeError("File: attempted to GetPosition() on non-seekable stream");
    return (uint64_t)hdfsTell(m_fs, m_file);
#else
    RuntimeError("File: CNTK was built without HDFS support.");
#endif
}

void hdfs_File::SetPosition(uint64_t pos)
{
#ifdef USE_HDFS
    if (!CanSeek())
        RuntimeError("File: attempted to SetPosition() on non-seekable stream");
    if (hdfsSeek(m_fs, m_file, pos) == HDFS_ERROR)
        RuntimeError("File: failed to seek at the given position %llu", pos);
#else
    RuntimeError("File: CNTK was built without HDFS support.");
#endif
}

void hdfs_File::GetLine(string& str)
{
    // It is inefficient to read files on HDFS line by line
#ifdef USE_HDFS
    BufferedReader br = new BufferedReader(new InputStreamReader())
#else
    RuntimeError("File: CNTK was built without HDFS support.");
#endif
}

template <typename String>
void GetLines(std::vector<String>& lines)
{
    // It is inefficient to read files on HDFS line by line
#ifdef USE_HDFS

#else
    RuntimeError("File: CNTK was built without HDFS support.");
#endif
}

template<class String>
static bool Exists(const String& filename)
{
    int loc = filename.find(":");
    filename = filename.substr(1);
    loc = filename.find("/");
    string path = filename.substr(loc + 1);

    return (hdfsExists(m_fs, path) == 0) ? true : false;
}


#endif

}}}
