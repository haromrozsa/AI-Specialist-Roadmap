package com.example.onnx;

import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.util.Set;

/**
 * Utility for validating user-supplied file paths against path traversal
 * and unintended-file-access vulnerabilities.
 *
 * Rules enforced:
 *  - Path must resolve to a real file on disk
 *  - Resolved real path must be inside the allowed base directory
 *  - Symlinks that escape the base directory are rejected
 *  - File extension must be in the allow-list
 */
@Slf4j
@UtilityClass
public class PathValidator {

    /**
     * Resolves and validates a user-supplied path.
     *
     * @param userPath      path supplied by the user (e.g. CLI arg)
     * @param baseDirectory directory the file MUST live inside
     * @param allowedExtensions allow-listed lowercase extensions (e.g. "onnx", "png")
     * @return the canonical, validated path
     * @throws SecurityException if the path is outside the base directory
     * @throws IOException       if the file does not exist or cannot be resolved
     */
    public static Path validate(Path userPath,
                                Path baseDirectory,
                                Set<String> allowedExtensions) throws IOException {

        if (userPath == null) {
            throw new IllegalArgumentException("Path must not be null");
        }

        Path baseReal = baseDirectory.toAbsolutePath().normalize().toRealPath();

        // Resolve relative paths against the base directory
        Path resolved = userPath.isAbsolute()
                ? userPath.normalize()
                : baseReal.resolve(userPath).normalize();

        if (!Files.exists(resolved, LinkOption.NOFOLLOW_LINKS)) {
            throw new IOException("File does not exist: " + resolved);
        }

        // Canonicalize (resolves symlinks and `..`)
        Path realPath = resolved.toRealPath();

        if (!realPath.startsWith(baseReal)) {
            log.warn("Rejected path outside base directory. base={} actual={}",
                    baseReal, realPath);
            throw new SecurityException(
                    "Path is outside the allowed directory: " + realPath);
        }

        if (!Files.isRegularFile(realPath)) {
            throw new IOException("Not a regular file: " + realPath);
        }

        String extension = extractExtension(realPath);
        if (!allowedExtensions.contains(extension)) {
            throw new SecurityException(
                    "Disallowed file extension: '" + extension
                            + "'. Allowed: " + allowedExtensions);
        }

        log.debug("Validated path: {}", realPath);
        return realPath;
    }

    private static String extractExtension(Path path) {
        String name = path.getFileName().toString();
        int dot = name.lastIndexOf('.');
        return (dot < 0) ? "" : name.substring(dot + 1).toLowerCase();
    }
}